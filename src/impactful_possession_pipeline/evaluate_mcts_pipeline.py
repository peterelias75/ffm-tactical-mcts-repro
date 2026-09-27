"""Paired MCTS-vs-no-intervention evaluation across a holdout of matches.

For each match, compares:
  - no_intervention: a single free-running rollout with no interventions at all
    (same mechanism as evaluate_pipeline.evaluate_match's no-op path)
  - mcts_planned: plan_second_half_mcts's PUCT-guided intervention sequence

Reports per-match delta (mcts possession - no-intervention possession), matching
the paper's "avg gain" / "% matches improved" reporting style, PLUS a bootstrap
LCB across the whole set for statistical rigor -- per this project's own
established discipline, naive %-improved/avg-gain alone is exactly the kind of
measurement that produced illusory findings before (see archive_phase2_v1).
"""

import argparse
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, MCTSConfig, PlannerConfig, TokenConfig
from .dataset import frames_after_cut, token_index_at_cut
from .decode import decode_frames, generate_continuation
from .mcts_planner import plan_second_half_mcts
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import bootstrap_mean_lcb, home_possession_ratio, home_shot_count, load_json, mean, save_json, std
from .vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--holdout-json", default="impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json")
    parser.add_argument("--noop-cache-json", default=None,
                        help="Optional completed compatible policy results whose per-match no_intervention_home_possession values are reused. Policy/intervention rollouts are always recomputed.")
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--num-simulations", type=int, default=MCTSConfig.num_simulations)
    parser.add_argument("--c-puct", type=float, default=MCTSConfig.c_puct)
    parser.add_argument("--selection-policy", choices=["visits", "mean", "risk_adjusted"], default=MCTSConfig.selection_policy)
    parser.add_argument("--risk-penalty", type=float, default=MCTSConfig.risk_penalty)
    parser.add_argument("--noop-margin", type=float, default=MCTSConfig.noop_margin)
    parser.add_argument("--reward-mode", choices=["possession", "paired_delta"], default=MCTSConfig.reward_mode)
    parser.add_argument("--shots-weight", type=float, default=MCTSConfig.shots_weight)
    parser.add_argument("--possession-objective-weight", type=float, default=MCTSConfig.possession_objective_weight)
    parser.add_argument("--uniform-prior-weight", type=float, default=MCTSConfig.uniform_prior_weight,
                         help="1.0 = plain MCTS (uniform priors, no learned-prior blend); paper default ~0.05 = priors-guided MCTS.")
    parser.add_argument("--dirichlet-epsilon", type=float, default=MCTSConfig.dirichlet_epsilon)
    parser.add_argument("--dirichlet-alpha", type=float, default=MCTSConfig.dirichlet_alpha)
    parser.add_argument("--allow-defensive-actions", action="store_true")
    parser.add_argument("--step-minutes", type=int, default=PlannerConfig.step_minutes)
    parser.add_argument("--intervention-cooldown-minutes", type=int, default=1,
                        help="After an active action at minute t, the next active action is allowed at t+N; decisions remain minute-by-minute.")
    parser.add_argument("--rollout-lookahead-tokens", type=int, default=None,
                         help="Literal max_new_tokens cap for each candidate-evaluation rollout. This never changes the real match advancement interval.")
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from --out-json if it already contains partial rows.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.84, device=0)
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    decode_cfg = DecodeConfig(temperature=args.temperature, top_p=args.top_p)
    planner_cfg = PlannerConfig(step_minutes=args.step_minutes, allow_defensive_actions=args.allow_defensive_actions)
    mcts_cfg = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        selection_policy=args.selection_policy,
        risk_penalty=args.risk_penalty,
        noop_margin=args.noop_margin,
        reward_mode=args.reward_mode,
        shots_weight=args.shots_weight,
        possession_objective_weight=args.possession_objective_weight,
        uniform_prior_weight=args.uniform_prior_weight,
        dirichlet_epsilon=args.dirichlet_epsilon,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    if args.rollout_lookahead_tokens is not None and args.rollout_lookahead_tokens < 0:
        parser.error("--rollout-lookahead-tokens must be non-negative")
    if args.intervention_cooldown_minutes < 1:
        parser.error("--intervention-cooldown-minutes must be at least 1")

    holdout = load_json(args.holdout_json)
    match_ids = holdout["match_ids"]
    if args.max_matches:
        match_ids = match_ids[: args.max_matches]

    noop_cache = {}
    if args.noop_cache_json:
        cache_data = load_json(args.noop_cache_json)
        cache_summary = cache_data.get("summary", {})
        if not cache_summary.get("complete"):
            raise ValueError(f"--noop-cache-json is not complete: {args.noop_cache_json}")
        cache_checkpoint = cache_summary.get("checkpoint")
        if cache_checkpoint and str(Path(cache_checkpoint).resolve()) != str(Path(args.checkpoint).resolve()):
            raise ValueError("--noop-cache-json checkpoint does not match --checkpoint")
        noop_cache = {
            str(row.get("match")): float(row["no_intervention_home_possession"])
            for row in cache_data.get("matches", [])
            if row.get("match") is not None and row.get("no_intervention_home_possession") is not None
        }
        missing = [str(match_id) for match_id in match_ids if str(match_id) not in noop_cache]
        if missing:
            raise ValueError(f"--noop-cache-json lacks {len(missing)} requested matches; first={missing[:3]}")
        print(f"validated no-op cache with {len(noop_cache)} matches from {args.noop_cache_json}", flush=True)

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
        summary = {
            "checkpoint": args.checkpoint,
            "holdout_json": args.holdout_json,
            "noop_cache_json": args.noop_cache_json,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_matches": len(rows),
            "num_requested_matches": len(match_ids),
            "num_simulations": args.num_simulations,
            "step_minutes": args.step_minutes,
            "intervention_cooldown_minutes": args.intervention_cooldown_minutes,
            "uniform_prior_weight": args.uniform_prior_weight,
            "rollout_lookahead_tokens": args.rollout_lookahead_tokens,
            "lookahead_minutes": None,
            "c_puct": args.c_puct,
            "selection_policy": args.selection_policy,
            "risk_penalty": args.risk_penalty,
            "noop_margin": args.noop_margin,
            "reward_mode": args.reward_mode,
            "shots_weight": args.shots_weight,
            "possession_objective_weight": args.possession_objective_weight,
            "mean_delta_pp": mean(deltas),
            "std_delta_pp": std(deltas),
            "pct_matches_improved": 100.0 * mean([1.0 if d > 0 else 0.0 for d in deltas]),
            "mean_gain_when_improved_pp": mean(positive) if positive else 0.0,
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

        if str(match_id) in noop_cache:
            noop_pos = noop_cache[str(match_id)]
            noop_source = args.noop_cache_json
        else:
            noop_generated = generate_continuation(
                model, vocab, prompt_ids, start_minute=args.cut_minute, end_minute=90,
                token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=seed,
            )
            noop_frames = decode_frames(vocab, noop_generated, start_minute=args.cut_minute)
            noop_pos = home_possession_ratio(noop_frames)
            noop_shots = home_shot_count(noop_frames)
            noop_source = "generated"

        if str(match_id) in noop_cache:
            # A possession-only cache cannot safely supply the shot baseline.
            noop_generated = generate_continuation(
                model, vocab, prompt_ids, start_minute=args.cut_minute, end_minute=90,
                token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=seed,
            )
            noop_frames = decode_frames(vocab, noop_generated, start_minute=args.cut_minute)
            noop_shots = home_shot_count(noop_frames)

        mcts_result = plan_second_half_mcts(
            model, vocab, prompt_ids, token_cfg=token_cfg, decode_cfg=decode_cfg,
            planner_cfg=planner_cfg, mcts_cfg=mcts_cfg, device=device,
            seed=seed, cut_minute=args.cut_minute, end_minute=90,
            lookahead_minutes=None,
            rollout_max_new_tokens=args.rollout_lookahead_tokens,
            intervention_cooldown_minutes=args.intervention_cooldown_minutes,
        )
        raw_mcts_pos = mcts_result["simulated_home_possession"]
        raw_mcts_shots = mcts_result["simulated_home_shots"]
        has_intervention = (mcts_result["used_subs"] + mcts_result["used_tactical_shifts"]) > 0
        mcts_pos = raw_mcts_pos if has_intervention else noop_pos
        mcts_shots = raw_mcts_shots if has_intervention else noop_shots
        delta = mcts_pos - noop_pos
        real_pos = home_possession_ratio(frames_after_cut(frames, args.cut_minute))
        real_shots = home_shot_count(frames_after_cut(frames, args.cut_minute))

        row = {
            "match": match_id,
            "real_home_possession": real_pos,
            "no_intervention_home_possession": noop_pos,
            "real_home_shots": real_shots,
            "no_intervention_home_shots": noop_shots,
            "no_intervention_source": noop_source,
            "mcts_home_possession": mcts_pos,
            "mcts_home_shots": mcts_shots,
            "shot_delta_vs_noop": mcts_shots - noop_shots,
            "shot_delta_vs_real": mcts_shots - real_shots,
            "possession_delta_vs_real_pp": 100.0 * (mcts_pos - real_pos),
            "delta_pp": delta * 100.0,
            "used_subs": mcts_result["used_subs"],
            "used_tactical_shifts": mcts_result["used_tactical_shifts"],
            "interventions": mcts_result["interventions"],
        }
        rows.append(row)
        done_matches.add(str(match_id))
        rows_changed = True
        write_state(complete=False)
        print(
            f"{i + 1}/{len(match_ids)} {match_id} noop={noop_pos:.3f} mcts={mcts_pos:.3f} "
            f"delta={delta * 100:+.2f}pp subs={row['used_subs']} tactics={row['used_tactical_shifts']}",
            flush=True,
        )

    write_state(complete=True)
    summary = load_json(args.out_json)["summary"]
    status = "saved" if rows_changed else "resaved"
    print(f"{status} to {args.out_json}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
