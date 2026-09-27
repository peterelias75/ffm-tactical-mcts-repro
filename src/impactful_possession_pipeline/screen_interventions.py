"""Full-holdout batched intervention screen using the deterministic value head.

This replaces the Phase 2 v1 pattern (archived in archive_phase2_v1/) of
hand-picking a few matches and chasing single-seed generation deltas. Every
legal home action is scored at every cut minute across the FULL pinned
diagnostic holdout using only `predict_values` (one deterministic forward
pass per context, no token generation, no seed noise), then aggregated with
a bootstrap lower confidence bound. Nothing here is reported as a finding
unless its LCB clears zero across the whole holdout -- not a handful of
hand-picked matches.

Usage:
    python3 -m impactful_possession_pipeline.screen_interventions \
        --checkpoint impactful_possession_pipeline/runs/<run>/best_forward_simulator.pt \
        --holdout-json impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json \
        --out-json impactful_possession_pipeline/runs/intervention_screen.json
"""

import argparse
from pathlib import Path

import torch

from .config import DataConfig, PlannerConfig, TokenConfig
from .dataset import token_index_at_cut
from .model import load_checkpoint
from .planner import home_actions, legal_actions, score_action
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import bootstrap_mean_lcb, load_json, mean, save_json, std
from .vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout-json", default="impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json")
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--cut-minutes", default="45,50,55,60,65")
    parser.add_argument("--max-holdout-matches", type=int, default=None, help="Truncate the pinned holdout for a quick smoke test.")
    parser.add_argument("--lcb-alpha", type=float, default=0.10, help="1 - confidence level for the reported LCB (0.10 = 90% CI).")
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/intervention_screen.json")
    args = parser.parse_args()

    cut_minutes = [int(x) for x in args.cut_minutes.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    planner_cfg = PlannerConfig()
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)

    holdout = load_json(args.holdout_json)
    match_ids = holdout["match_ids"]
    if args.max_holdout_matches:
        match_ids = match_ids[: args.max_holdout_matches]
    print(f"screening {len(match_ids)} holdout matches x {len(cut_minutes)} cut minutes x actions", flush=True)

    per_action_deltas: dict[str, list[float]] = {}
    per_action_utility: dict[str, list[float]] = {}
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
            candidates = legal_actions(vocab, used_subs=0, used_tactics=0, cooldown=0, cfg=planner_cfg, token_cfg=token_cfg)
            for action in candidates:
                if action == "ACTION::NOOP":
                    continue
                score = score_action(model, vocab, prompt_ids, action, planner_cfg, device)
                per_action_deltas.setdefault(action, []).append(score.delta)
                per_action_utility.setdefault(action, []).append(score.utility)
                rows.append({
                    "match": match_id,
                    "cut_minute": cut_minute,
                    "action": action,
                    "delta": score.delta,
                    "uncertainty": score.uncertainty,
                    "utility": score.utility,
                })
        if (mi + 1) % 5 == 0 or mi == len(match_ids) - 1:
            print(f"  screened {mi + 1}/{len(match_ids)} matches", flush=True)

    summary = []
    for action, deltas in per_action_deltas.items():
        lcb = bootstrap_mean_lcb(deltas, alpha=args.lcb_alpha, seed=hash(action) % (2**31))
        summary.append({
            "action": action,
            "n": len(deltas),
            "mean_delta_pp": mean(deltas) * 100.0,
            "std_delta_pp": std(deltas) * 100.0,
            "lcb_pp": lcb * 100.0,
            "positive_rate": mean([1.0 if d > 0 else 0.0 for d in deltas]),
            "passes_full_holdout_lcb": lcb > 0.0,
        })
    summary.sort(key=lambda r: r["lcb_pp"], reverse=True)

    out = {
        "checkpoint": args.checkpoint,
        "holdout_json": args.holdout_json,
        "num_holdout_matches": len(match_ids),
        "cut_minutes": cut_minutes,
        "lcb_alpha": args.lcb_alpha,
        "action_summary": summary,
        "rows": rows,
    }
    save_json(out, args.out_json)
    print(f"saved intervention screen to {args.out_json}", flush=True)
    print("ranked actions (by full-holdout LCB):", flush=True)
    for r in summary:
        flag = "PASSES" if r["passes_full_holdout_lcb"] else "no"
        print(f"  {r['action']:35s} n={r['n']:4d} mean={r['mean_delta_pp']:+6.2f}pp lcb={r['lcb_pp']:+6.2f}pp {flag}", flush=True)


if __name__ == "__main__":
    main()
