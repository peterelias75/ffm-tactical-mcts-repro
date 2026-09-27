import argparse
import math
from pathlib import Path

import torch

from .config import DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .evaluate_sequential_plan import replay
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, mean, save_json, std
from .vocabulary import Vocab


DEFAULT_ACTIONS = [
    "ACTION::HOME_SUB_ROLE_CENTRAL_MIDFIELDER",
    "ACTION::HOME_SUB_ROLE_WINGER",
    "ACTION::HOME_SUB_ROLE_FORWARD",
    "ACTION::HOME_TACTIC_FORM_433",
]


def lcb(values: list[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def summarize(rows: list[dict], z: float) -> dict:
    deltas = [row["delta_pp"] for row in rows]
    return {
        "n": len(deltas),
        "mean_delta_pp": mean(deltas),
        "std_delta_pp": std(deltas),
        "lcb_delta_pp": lcb(deltas, z),
        "positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in deltas]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Risk-aware multi-action selector using paired rollout lower-confidence bounds."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--match-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--actions", nargs="*", default=DEFAULT_ACTIONS)
    parser.add_argument("--rollout-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=50)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--lcb-z", type=float, default=1.64)
    parser.add_argument("--min-lcb-pp", type=float, default=0.0)
    parser.add_argument("--min-selection-seeds", type=int, default=8)
    parser.add_argument("--stop-action-if-delta-below", type=float, default=None)
    parser.add_argument("--min-action-seeds-before-stop", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    decode_cfg = DecodeConfig()

    events = load_json(args.match_json)
    ids, frames = encoder.encode_match(events, None)
    prompt_ids = ids[: token_index_at_cut(frames, args.cut_minute)]
    n_steps = max(1, (args.end_minute - args.cut_minute) // args.step_minutes)
    noop_actions = ["ACTION::NOOP"] * n_steps
    policy_by_action = {
        action: [action] + ["ACTION::NOOP"] * (n_steps - 1)
        for action in args.actions
    }

    rows_by_action = {action: [] for action in args.actions}
    stopped_actions: dict[str, dict] = {}
    noop_by_seed = {}
    for seed_index in range(args.rollout_seeds):
        seed = args.seed + seed_index * 1000
        noop = replay(
            model,
            vocab,
            prompt_ids,
            noop_actions,
            args.cut_minute,
            args.step_minutes,
            token_cfg,
            decode_cfg,
            device,
            seed,
        )
        noop_by_seed[str(seed)] = noop
        for action, policy_actions in policy_by_action.items():
            if action in stopped_actions:
                print(
                    f"[skip] seed={seed} action={action} "
                    f"reason={stopped_actions[action]['reason']}",
                    flush=True,
                )
                continue
            policy = replay(
                model,
                vocab,
                prompt_ids,
                policy_actions,
                args.cut_minute,
                args.step_minutes,
                token_cfg,
                decode_cfg,
                device,
                seed,
            )
            row = {
                "seed": seed,
                "noop_possession": noop,
                "policy_possession": policy,
                "delta_pp": (policy - noop) * 100.0,
            }
            rows_by_action[action].append(row)
            print(
                f"[progress] seed={seed} action={action} delta_pp={row['delta_pp']:.3f}",
                flush=True,
            )
            if (
                args.stop_action_if_delta_below is not None
                and len(rows_by_action[action]) >= args.min_action_seeds_before_stop
                and row["delta_pp"] <= args.stop_action_if_delta_below
            ):
                stopped_actions[action] = {
                    "reason": f"delta_pp <= {args.stop_action_if_delta_below}",
                    "seed": seed,
                    "delta_pp": row["delta_pp"],
                    "completed_action_seeds": len(rows_by_action[action]),
                }
                print(
                    f"[early_stop_action] seed={seed} action={action} "
                    f"delta_pp={row['delta_pp']:.3f} "
                    f"threshold={args.stop_action_if_delta_below:.3f}",
                    flush=True,
                )

        partial_ranked = sorted(
            (
                {
                    "action": action,
                    **summarize(rows, args.lcb_z),
                    "rows": rows,
                }
                for action, rows in rows_by_action.items()
                if rows
            ),
            key=lambda item: (item["lcb_delta_pp"], item["mean_delta_pp"]),
            reverse=True,
        )
        save_json(
            {
                "complete": False,
                "checkpoint": args.checkpoint,
                "match_json": args.match_json,
                "match": Path(args.match_json).stem,
                "cut_minute": args.cut_minute,
                "end_minute": args.end_minute,
                "rollout_seeds": args.rollout_seeds,
                "completed_seeds": seed_index + 1,
                "stopped_actions": stopped_actions,
                "lcb_z": args.lcb_z,
                "min_lcb_pp": args.min_lcb_pp,
                "noop_by_seed": noop_by_seed,
                "ranked_actions": partial_ranked,
            },
            args.out_json,
        )

    ranked = sorted(
        (
            {
                "action": action,
                **summarize(rows, args.lcb_z),
                "rows": rows,
            }
            for action, rows in rows_by_action.items()
        ),
        key=lambda item: (item["lcb_delta_pp"], item["mean_delta_pp"]),
        reverse=True,
    )
    eligible = [item for item in ranked if item["lcb_delta_pp"] >= args.min_lcb_pp]
    provisional_selected = eligible[0] if eligible else None
    enough_seeds = args.rollout_seeds >= args.min_selection_seeds
    selected = provisional_selected if enough_seeds else None
    selection_status = "selected" if selected else "no_eligible_action"
    if not enough_seeds:
        selection_status = "insufficient_seeds"
    save_json(
        {
            "complete": True,
            "checkpoint": args.checkpoint,
            "match_json": args.match_json,
            "match": Path(args.match_json).stem,
            "cut_minute": args.cut_minute,
            "end_minute": args.end_minute,
            "rollout_seeds": args.rollout_seeds,
            "lcb_z": args.lcb_z,
            "min_lcb_pp": args.min_lcb_pp,
            "min_selection_seeds": args.min_selection_seeds,
            "stopped_actions": stopped_actions,
            "selection_status": selection_status,
            "selected_action": selected["action"] if selected else "ACTION::NOOP",
            "selected": selected,
            "provisional_selected_action": provisional_selected["action"] if provisional_selected else "ACTION::NOOP",
            "provisional_selected": provisional_selected,
            "noop_by_seed": noop_by_seed,
            "ranked_actions": ranked,
        },
        args.out_json,
    )


if __name__ == "__main__":
    main()
