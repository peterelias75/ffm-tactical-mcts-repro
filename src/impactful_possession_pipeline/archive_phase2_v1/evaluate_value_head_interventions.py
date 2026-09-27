import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch

from .config import DataConfig, TokenConfig, action_tokens_for_config
from .dataset import token_index_at_cut
from .model import load_checkpoint
from .planner import action_kind
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, mean, save_json
from .vocabulary import Vocab


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict_rest_value(model, ids: Sequence[int], device: torch.device) -> tuple[float, float]:
    context = torch.tensor([list(ids)], dtype=torch.long, device=device)
    mask = torch.ones_like(context, dtype=torch.bool)
    pred = model.predict_values(context, mask)
    rest_pos = float(pred["possession"][0, 2].detach().cpu())
    rest_unc = float(pred["uncertainty"][0, 2].detach().cpu())
    return rest_pos, rest_unc


def candidate_actions(vocab: Vocab, token_cfg: TokenConfig, allow_defensive: bool) -> list[str]:
    actions = []
    for action in action_tokens_for_config(token_cfg):
        if not action.startswith("ACTION::HOME_"):
            continue
        if not allow_defensive and action.endswith("_DEFEND"):
            continue
        if action in vocab.stoi:
            actions.append(action)
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate value-head intervention deltas on checkpoint holdout matches.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/value_head_interventions.json")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all checkpoint test files.")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--intervention-cost", type=float, default=0.004)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.15)
    parser.add_argument("--allow-defensive", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    if "test_files" not in extra or "data_cfg" not in extra:
        raise RuntimeError("Checkpoint does not include test_files/data_cfg metadata.")

    vocab = Vocab.from_dict(vocab_obj)
    data_cfg = DataConfig(**extra["data_cfg"])
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    files = list(extra["test_files"])
    if args.max_files > 0:
        files = files[: args.max_files]

    actions = candidate_actions(vocab, token_cfg, args.allow_defensive)
    rows = []
    action_counts = Counter()
    for event_file in files:
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, data_cfg.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        prompt_ids = ids[: token_index_at_cut(frames, args.cut_minute)]
        base_pos, base_unc = predict_rest_value(model, prompt_ids, device)

        scores = []
        for action in actions:
            action_pos, action_unc = predict_rest_value(model, list(prompt_ids) + [vocab.id(action)], device)
            delta = action_pos - base_pos
            utility = delta - args.intervention_cost - args.uncertainty_penalty * action_unc
            scores.append(
                {
                    "action": action,
                    "kind": action_kind(action),
                    "predicted_possession": action_pos,
                    "uncertainty": action_unc,
                    "delta": delta,
                    "utility": utility,
                }
            )
        scores.sort(key=lambda row: row["utility"], reverse=True)
        best = scores[0] if scores else None
        chosen = "ACTION::NOOP"
        accepted = False
        if best and best["delta"] >= args.min_delta and best["utility"] > 0:
            chosen = best["action"]
            accepted = True
        action_counts[chosen] += 1
        rows.append(
            {
                "match": Path(event_file).name,
                "cut_minute": args.cut_minute,
                "base_predicted_possession": base_pos,
                "base_uncertainty": base_unc,
                "chosen_action": chosen,
                "accepted": accepted,
                "best_delta": best["delta"] if best else 0.0,
                "best_utility": best["utility"] if best else 0.0,
                "scores": scores,
            }
        )

    accepted_rows = [row for row in rows if row["accepted"]]
    summary = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "holdout_files": len(files),
        "cut_minute": args.cut_minute,
        "actions_considered": actions,
        "min_delta": args.min_delta,
        "intervention_cost": args.intervention_cost,
        "uncertainty_penalty": args.uncertainty_penalty,
        "accept_rate": len(accepted_rows) / max(1, len(rows)),
        "mean_base_predicted_possession": mean([row["base_predicted_possession"] for row in rows]),
        "mean_best_delta_pp": mean([row["best_delta"] * 100.0 for row in rows]),
        "mean_best_utility_pp": mean([row["best_utility"] * 100.0 for row in rows]),
        "mean_accepted_delta_pp": mean([row["best_delta"] * 100.0 for row in accepted_rows]),
        "chosen_action_counts": dict(action_counts.most_common()),
    }
    save_json({"summary": summary, "matches": rows}, args.out_json)
    print(f"saved value-head intervention evaluation to {args.out_json}")
    print(summary)


if __name__ == "__main__":
    main()
