"""Generation-based confirmation of ONE candidate action across a holdout fold.

screen_interventions.py's deterministic value-head screen is the primary search
mechanism (cheap, no seed noise, safe to run over many actions/cut-minutes). This
script is the independent second check for whichever action(s) clear that
pre-registered screen on the search fold: real paired no-op-vs-action free-running
rollouts (planner.confirm_action), aggregated with a bootstrap LCB across every
(match, cut-minute) pair in the fold -- not a hand-picked match or a single seed,
which is what let v1's "confirmed" corridors turn out to be noise.

Usage:
    python3 -m impactful_possession_pipeline.confirm_intervention_batch \
        --checkpoint <path> --action ACTION::HOME_SUB_ROLE_FORWARD \
        --holdout-json impactful_possession_pipeline/runs/pinned_holdout_fold_confirm_12.json \
        --temperature 0.80 --top-p 0.95 --seeds 12 \
        --out-json impactful_possession_pipeline/runs/full_holdout_checks/confirm_HOME_SUB_ROLE_FORWARD.json
"""

import argparse
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, PlannerConfig, TokenConfig
from .dataset import token_index_at_cut
from .model import load_checkpoint
from .planner import confirm_action
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import bootstrap_mean_lcb, load_json, mean, save_json, std
from .vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action", required=True, help="e.g. ACTION::HOME_SUB_ROLE_FORWARD")
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--holdout-json", default="impactful_possession_pipeline/runs/pinned_holdout_fold_confirm_12.json")
    parser.add_argument("--cut-minutes", default="45,50,55,60,65")
    parser.add_argument("--seeds", type=int, default=12, help="Rollout seeds per (match, cut-minute) pair -- fewer than the planner's default confirm_rollout_seeds since this multiplies across the whole fold.")
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--lcb-alpha", type=float, default=0.10)
    parser.add_argument("--base-seed", type=int, default=20260722)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    cut_minutes = [int(x) for x in args.cut_minutes.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    decode_cfg = DecodeConfig(temperature=args.temperature, top_p=args.top_p)
    planner_cfg = PlannerConfig()

    holdout = load_json(args.holdout_json)
    match_ids = holdout["match_ids"]
    print(f"confirming {args.action} across {len(match_ids)} matches x {len(cut_minutes)} cut minutes x {args.seeds} seeds", flush=True)

    all_deltas = []
    rows = []
    for mi, match_id in enumerate(match_ids):
        event_file = str(Path(args.events_dir) / match_id)
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, args.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        for cut_minute in cut_minutes:
            cut_idx = token_index_at_cut(frames, cut_minute)
            if cut_idx <= 2 or cut_idx >= len(ids) - 2:
                continue
            prompt_ids = ids[:cut_idx]
            result = confirm_action(
                model, vocab, prompt_ids, cut_minute, args.action,
                token_cfg, decode_cfg, planner_cfg, device,
                base_seed=args.base_seed + mi * 1000 + cut_minute,
                num_seeds=args.seeds,
            )
            all_deltas.extend(result.deltas)
            rows.append({
                "match": match_id,
                "cut_minute": cut_minute,
                "mean_delta_pp": result.mean_delta * 100.0,
                "lcb_pp": result.lcb * 100.0,
                "positive_seed_rate": result.positive_seed_rate,
            })
            print(f"  {match_id}@{cut_minute} mean_delta={result.mean_delta*100:+.2f}pp lcb={result.lcb*100:+.2f}pp positive_rate={result.positive_seed_rate:.2f}", flush=True)

    overall_lcb = bootstrap_mean_lcb(all_deltas, alpha=args.lcb_alpha, seed=args.base_seed)
    summary = {
        "checkpoint": args.checkpoint,
        "action": args.action,
        "holdout_json": args.holdout_json,
        "cut_minutes": cut_minutes,
        "seeds_per_pair": args.seeds,
        "n_total_rollout_pairs": len(all_deltas),
        "mean_delta_pp": mean(all_deltas) * 100.0,
        "std_delta_pp": std(all_deltas) * 100.0,
        "lcb_pp": overall_lcb * 100.0,
        "positive_rate": mean([1.0 if d > 0 else 0.0 for d in all_deltas]),
        "passes_confirmation": overall_lcb > 0.0,
    }
    save_json({"summary": summary, "rows": rows}, args.out_json)
    print(f"saved to {args.out_json}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
