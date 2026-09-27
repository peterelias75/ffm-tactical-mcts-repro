"""Evaluate value-head shortlist planner with paired rollout confirmation.

This is a faster alternative to paired-delta MCTS: score all legal actions with the
value head, confirm only the top candidate(s) with paired same-seed rollouts, and
fall back to NOOP unless confirmation clears a bootstrap lower bound.
"""

import argparse
import time
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, PlannerConfig, TokenConfig, action_tokens_for_config
from .dataset import frames_after_cut, token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .planner import plan_second_half
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import bootstrap_mean_lcb, home_possession_ratio, load_json, mean, save_json, std
from .vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--holdout-json", default="impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json")
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--confirm-top-k", type=int, default=1)
    parser.add_argument("--confirm-rollout-seeds", type=int, default=3)
    parser.add_argument("--confirm-min-lcb", type=float, default=PlannerConfig.confirm_min_lcb)
    parser.add_argument("--confirm-min-positive-seed-rate", type=float, default=PlannerConfig.confirm_min_positive_seed_rate)
    parser.add_argument("--confirm-horizon", choices=["step", "rest"], default=PlannerConfig.confirm_horizon)
    parser.add_argument("--final-confirm-horizon", choices=["none", "step", "rest"], default=PlannerConfig.final_confirm_horizon)
    parser.add_argument("--final-confirm-rollout-seeds", type=int, default=PlannerConfig.final_confirm_rollout_seeds)
    parser.add_argument("--final-confirm-min-lcb", type=float, default=PlannerConfig.final_confirm_min_lcb)
    parser.add_argument("--final-confirm-min-positive-seed-rate", type=float, default=PlannerConfig.final_confirm_min_positive_seed_rate)
    parser.add_argument("--final-confirm-all-shortlist", action="store_true")
    parser.add_argument("--min-delta-gate", type=float, default=0.015)
    parser.add_argument("--risk-penalty", type=float, default=0.35)
    parser.add_argument("--intervention-cost", type=float, default=0.004)
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--exclude-action", action="append", default=[])
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    decode_cfg = DecodeConfig(temperature=args.temperature, top_p=args.top_p)
    excluded_actions = set(args.exclude_action)
    allowed_actions = None
    if excluded_actions:
        allowed_actions = [
            action for action in action_tokens_for_config(token_cfg)
            if action.startswith("ACTION::HOME_") and action not in excluded_actions
        ]

    planner_cfg = PlannerConfig(
        step_minutes=args.step_minutes,
        confirm_top_k=args.confirm_top_k,
        confirm_rollout_seeds=args.confirm_rollout_seeds,
        confirm_min_lcb=args.confirm_min_lcb,
        confirm_min_positive_seed_rate=args.confirm_min_positive_seed_rate,
        confirm_horizon=args.confirm_horizon,
        final_confirm_horizon=args.final_confirm_horizon,
        final_confirm_rollout_seeds=args.final_confirm_rollout_seeds,
        final_confirm_min_lcb=args.final_confirm_min_lcb,
        final_confirm_min_positive_seed_rate=args.final_confirm_min_positive_seed_rate,
        final_confirm_all_shortlist=args.final_confirm_all_shortlist,
        min_delta_gate=args.min_delta_gate,
        risk_penalty=args.risk_penalty,
        intervention_cost=args.intervention_cost,
    )

    holdout = load_json(args.holdout_json)
    match_ids = holdout["match_ids"]
    if args.max_matches:
        match_ids = match_ids[: args.max_matches]

    out_path = Path(args.out_json)
    rows = []
    done_matches = set()
    if args.resume and out_path.exists():
        partial = load_json(out_path)
        rows = list(partial.get("matches", []))
        done_matches = {str(r.get("match")) for r in rows}
        print(f"resuming from {out_path}: {len(done_matches)} completed matches", flush=True)

    def write_state(complete: bool) -> None:
        deltas = [r["delta_pp"] for r in rows]
        positive = [d for d in deltas if d > 0]
        confirmation_counts = [r["num_confirmations"] for r in rows]
        intervention_counts = [r["num_interventions"] for r in rows]
        elapsed_seconds = [r.get("elapsed_seconds", 0.0) for r in rows]
        summary = {
            "checkpoint": args.checkpoint,
            "num_matches": len(rows),
            "num_requested_matches": len(match_ids),
            "planner": "value_head_shortlist_paired_confirmation",
            "cut_minute": args.cut_minute,
            "step_minutes": args.step_minutes,
            "confirm_top_k": args.confirm_top_k,
            "confirm_rollout_seeds": args.confirm_rollout_seeds,
            "confirm_min_lcb": args.confirm_min_lcb,
            "confirm_min_positive_seed_rate": args.confirm_min_positive_seed_rate,
            "confirm_horizon": args.confirm_horizon,
            "final_confirm_horizon": args.final_confirm_horizon,
            "final_confirm_rollout_seeds": args.final_confirm_rollout_seeds,
            "final_confirm_min_lcb": args.final_confirm_min_lcb,
            "final_confirm_min_positive_seed_rate": args.final_confirm_min_positive_seed_rate,
            "final_confirm_all_shortlist": args.final_confirm_all_shortlist,
            "min_delta_gate": args.min_delta_gate,
            "risk_penalty": args.risk_penalty,
            "intervention_cost": args.intervention_cost,
            "excluded_actions": sorted(excluded_actions),
            "mean_delta_pp": mean(deltas),
            "std_delta_pp": std(deltas),
            "pct_matches_improved": 100.0 * mean([1.0 if d > 0 else 0.0 for d in deltas]),
            "mean_gain_when_improved_pp": mean(positive) if positive else 0.0,
            "mean_interventions_per_match": mean(intervention_counts),
            "mean_confirmations_per_match": mean(confirmation_counts),
            "mean_elapsed_seconds_per_match": mean(elapsed_seconds),
            "bootstrap_lcb90_pp": bootstrap_mean_lcb(deltas, alpha=0.10, seed=args.seed) if deltas else 0.0,
            "complete": complete,
        }
        summary["passes_lcb"] = summary["bootstrap_lcb90_pp"] > 0.0
        save_json({"summary": summary, "matches": rows}, args.out_json)

    rows_changed = False
    for i, match_id in enumerate(match_ids):
        if str(match_id) in done_matches:
            print(f"{i + 1}/{len(match_ids)} {match_id} already done; skipping", flush=True)
            continue
        match_start = time.time()
        event_file = str(Path(args.events_dir) / match_id)
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, args.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        cut_idx = token_index_at_cut(frames, args.cut_minute)
        if cut_idx <= 2 or cut_idx >= len(ids) - 2:
            print(f"{i + 1}/{len(match_ids)} {match_id} skipped (cut index out of range)", flush=True)
            continue
        prompt_ids = ids[:cut_idx]
        seed = args.seed + i

        noop_generated = generate_continuation(
            model, vocab, prompt_ids, start_minute=args.cut_minute, end_minute=90,
            token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=seed,
        )
        noop_frames = decode_frames(vocab, noop_generated, start_minute=args.cut_minute)
        noop_pos = home_possession_ratio(noop_frames)

        plan = plan_second_half(
            model, vocab, prompt_ids, token_cfg=token_cfg, decode_cfg=decode_cfg,
            planner_cfg=planner_cfg, device=device, seed=seed,
            cut_minute=args.cut_minute, end_minute=90, allowed_actions=allowed_actions, require_confirmation=True,
        )
        raw_plan_pos = plan["simulated_home_possession"]
        real_pos = home_possession_ratio(frames_after_cut(frames, args.cut_minute))
        chosen = [x["chosen_action"] for x in plan["interventions"] if x["chosen_action"] != "ACTION::NOOP"]
        confirmations = sum(len(x.get("confirmations", [])) for x in plan["interventions"])
        plan_pos = raw_plan_pos if chosen else noop_pos
        delta = plan_pos - noop_pos

        elapsed = time.time() - match_start

        row = {
            "match": match_id,
            "real_home_possession": real_pos,
            "no_intervention_home_possession": noop_pos,
            "planned_home_possession": plan_pos,
            "delta_pp": delta * 100.0,
            "num_interventions": len(chosen),
            "num_confirmations": confirmations,
            "elapsed_seconds": elapsed,
            "used_subs": plan["used_subs"],
            "used_tactical_shifts": plan["used_tactical_shifts"],
            "interventions": plan["interventions"],
        }
        rows.append(row)
        rows_changed = True
        write_state(complete=False)
        print(
            f"{i + 1}/{len(match_ids)} {match_id} noop={noop_pos:.3f} plan={plan_pos:.3f} "
            f"delta={delta * 100:+.2f}pp interventions={len(chosen)} confirmations={confirmations} elapsed={elapsed:.1f}s",
            flush=True,
        )

    write_state(complete=True)
    summary = load_json(args.out_json)["summary"]
    status = "saved" if rows_changed else "resaved"
    print(f"{status} to {args.out_json}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
