import argparse
import time
from collections import Counter
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, PlannerConfig, TokenConfig
from .dataset import frames_after_cut, token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .planner import plan_second_half
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, load_json, mean, save_json
from .vocabulary import Vocab


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_decode_cfg(args, frames: list[dict], cut_minute: int, conditional_prior_cfg: dict | None = None) -> DecodeConfig:
    target_events_per_minute = args.target_events_per_minute
    if args.density_from_prefix:
        lookback_start = max(0, cut_minute - args.density_lookback_minutes)
        prefix_frames = [frame for frame in frames if lookback_start <= frame.get("minute", 0) < cut_minute]
        target_events_per_minute = len(prefix_frames) / max(1, cut_minute - lookback_start)
    if target_events_per_minute is not None:
        target_events_per_minute *= max(0.1, float(args.density_multiplier))

    target_home_possession = None
    if args.prefix_possession_target:
        lookback_start = max(0, cut_minute - args.possession_lookback_minutes)
        prefix_frames = [frame for frame in frames if lookback_start <= frame.get("minute", 0) < cut_minute]
        target_home_possession = home_possession_ratio(prefix_frames)

    return DecodeConfig(
        top_p=args.top_p,
        temperature=args.temperature,
        prompt_tail_tokens=args.prompt_tail_tokens,
        target_events_per_minute=target_events_per_minute,
        event_density_tolerance=args.event_density_tolerance,
        target_home_possession=target_home_possession,
        external_possession_weight=args.external_possession_weight,
        possession_guidance_strength=args.possession_guidance_strength,
        conditional_prior=conditional_prior_cfg,
        conditional_prior_bias_strength=args.conditional_prior_bias_strength,
        enforce_possession_state=args.enforce_possession_state,
        football_structure_constraints=args.football_structure_constraints,
        pass_receipt_boost=args.pass_receipt_boost,
        pass_carry_boost=args.pass_carry_boost,
        overproduced_event_penalty=args.overproduced_event_penalty,
        max_pass_run=args.max_pass_run,
        force_receipt_after_pass_run=args.force_receipt_after_pass_run,
    )


def build_planner_summary(args, rows: list[dict], action_counts: Counter, checkpoint: str, device: torch.device, complete: bool) -> dict:
    return {
        "checkpoint": checkpoint,
        "device": str(device),
        "offset": args.offset,
        "num_matches": len(rows),
        "complete": complete,
        "updated_at_unix": time.time(),
        "decode_recipe": {
            "density_multiplier": args.density_multiplier,
            "possession_guidance_strength": args.possession_guidance_strength,
            "external_possession_weight": args.external_possession_weight,
            "conditional_prior_json": args.conditional_prior_json,
            "conditional_prior_bias_strength": args.conditional_prior_bias_strength,
            "football_structure_constraints": args.football_structure_constraints,
            "pass_receipt_boost": args.pass_receipt_boost,
            "pass_carry_boost": args.pass_carry_boost,
            "overproduced_event_penalty": args.overproduced_event_penalty,
        },
        "planner_recipe": {
            "rollout_seeds": args.rollout_seeds,
            "step_minutes": args.step_minutes,
            "min_delta_gate": args.min_delta_gate,
            "intervention_cost": args.intervention_cost,
            "risk_penalty": args.risk_penalty,
            "uncertainty_penalty": args.uncertainty_penalty,
            "only_actions": [a.strip() for a in args.only_actions.split(",") if a.strip()],
        },
        "mean_real_home_possession": mean([row["real_home_possession"] for row in rows]),
        "mean_noop_home_possession": mean([row["noop_home_possession"] for row in rows]),
        "mean_planned_home_possession": mean([row["planned_home_possession"] for row in rows]),
        "mean_planned_delta_vs_noop_pp": mean([row["planned_delta_vs_noop_pp"] for row in rows]),
        "mean_noop_abs_error_pp": mean([row["noop_abs_error_pp"] for row in rows]),
        "mean_planned_abs_error_vs_real_pp": mean([row["planned_abs_error_vs_real_pp"] for row in rows]),
        "paper_metric_mean_planned_minus_real_pp": mean([(row["planned_home_possession"] - row["real_home_possession"]) * 100.0 for row in rows]),
        "paper_metric_improvement_rate_vs_real": mean([1.0 if row["planned_home_possession"] > row["real_home_possession"] else 0.0 for row in rows]),
        "intervention_rate": mean([1.0 if row["num_interventions"] > 0 else 0.0 for row in rows]),
        "mean_interventions_per_match": mean([row["num_interventions"] for row in rows]),
        "chosen_action_counts": dict(action_counts.most_common()),
    }


def save_planner_progress(args, rows: list[dict], action_counts: Counter, checkpoint: str, device: torch.device, complete: bool) -> None:
    save_json({"summary": build_planner_summary(args, rows, action_counts, checkpoint, device, complete), "matches": rows}, args.out_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired no-op vs planned intervention rollouts on checkpoint holdout matches.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout-seeds", type=int, default=PlannerConfig.rollout_seeds)
    parser.add_argument("--step-minutes", type=int, default=PlannerConfig.step_minutes)
    parser.add_argument("--min-delta-gate", type=float, default=PlannerConfig.min_delta_gate)
    parser.add_argument("--intervention-cost", type=float, default=PlannerConfig.intervention_cost)
    parser.add_argument("--risk-penalty", type=float, default=PlannerConfig.risk_penalty)
    parser.add_argument("--uncertainty-penalty", type=float, default=PlannerConfig.uncertainty_penalty)
    parser.add_argument("--allow-defensive-actions", action="store_true")
    parser.add_argument("--only-actions", default="", help="Comma-separated action tokens the planner may choose, excluding NOOP.")
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=512)
    parser.add_argument("--possession-guidance-strength", type=float, default=DecodeConfig.possession_guidance_strength)
    parser.add_argument("--target-events-per-minute", type=float, default=None)
    parser.add_argument("--density-from-prefix", action="store_true")
    parser.add_argument("--density-lookback-minutes", type=int, default=15)
    parser.add_argument("--density-multiplier", type=float, default=1.12)
    parser.add_argument("--event-density-tolerance", type=int, default=DecodeConfig.event_density_tolerance)
    parser.add_argument("--prefix-possession-target", action="store_true")
    parser.add_argument("--possession-lookback-minutes", type=int, default=15)
    parser.add_argument("--external-possession-weight", type=float, default=0.75)
    parser.add_argument("--conditional-prior-json", default=None)
    parser.add_argument("--conditional-prior-bias-strength", type=float, default=0.0)
    parser.add_argument("--enforce-possession-state", action="store_true")
    parser.add_argument("--football-structure-constraints", action="store_true")
    parser.add_argument("--pass-receipt-boost", type=float, default=1.0)
    parser.add_argument("--pass-carry-boost", type=float, default=0.4)
    parser.add_argument("--overproduced-event-penalty", type=float, default=0.5)
    parser.add_argument("--max-pass-run", type=int, default=4)
    parser.add_argument("--force-receipt-after-pass-run", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    device = resolve_device(args.device)
    conditional_prior_cfg = load_json(args.conditional_prior_json) if args.conditional_prior_json else {}
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    if "test_files" not in extra or "data_cfg" not in extra:
        raise RuntimeError("Checkpoint does not include test_files/data_cfg metadata.")

    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    data_cfg = DataConfig(**extra["data_cfg"])
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    planner_cfg = PlannerConfig(
        step_minutes=args.step_minutes,
        rollout_seeds=args.rollout_seeds,
        min_delta_gate=args.min_delta_gate,
        intervention_cost=args.intervention_cost,
        risk_penalty=args.risk_penalty,
        uncertainty_penalty=args.uncertainty_penalty,
        allow_defensive_actions=args.allow_defensive_actions,
    )

    rows = []
    action_counts = Counter()
    all_files = list(extra["test_files"])
    files = all_files[args.offset:]
    if args.max_files > 0:
        files = files[: args.max_files]
    started = time.time()
    for idx, event_file in enumerate(files):
        print("[progress] planner match {}/{} file={}".format(idx + 1, len(files), Path(event_file).name), flush=True)
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, data_cfg.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        cut_idx = token_index_at_cut(frames, args.cut_minute)
        prompt_ids = ids[:cut_idx]
        decode_cfg = build_decode_cfg(args, frames, args.cut_minute, conditional_prior_cfg)
        real_frames = frames_after_cut(frames, args.cut_minute, args.end_minute - args.cut_minute)
        real_possession = home_possession_ratio(real_frames)

        noop_generated = generate_continuation(
            model,
            vocab,
            prompt_ids,
            start_minute=args.cut_minute,
            end_minute=args.end_minute,
            token_cfg=token_cfg,
            decode_cfg=decode_cfg,
            device=device,
            seed=args.seed + idx,
        )
        noop_frames = decode_frames(vocab, noop_generated, start_minute=args.cut_minute)
        noop_possession = home_possession_ratio(noop_frames)

        plan = plan_second_half(
            model,
            vocab,
            prompt_ids,
            token_cfg=token_cfg,
            decode_cfg=decode_cfg,
            planner_cfg=planner_cfg,
            device=device,
            seed=args.seed + idx,
            end_minute=args.end_minute,
            allowed_actions=[a.strip() for a in args.only_actions.split(",") if a.strip()] or None,
        )
        planned_possession = plan["simulated_home_possession"]
        chosen = [step["chosen_action"] for step in plan["interventions"] if step["chosen_action"] != "ACTION::NOOP"]
        action_counts.update(chosen)
        rows.append(
            {
                "match": Path(event_file).name,
                "real_home_possession": real_possession,
                "noop_home_possession": noop_possession,
                "planned_home_possession": planned_possession,
                "planned_delta_vs_noop_pp": (planned_possession - noop_possession) * 100.0,
                "noop_abs_error_pp": abs(noop_possession - real_possession) * 100.0,
                "planned_abs_error_vs_real_pp": abs(planned_possession - real_possession) * 100.0,
                "num_interventions": len(chosen),
                "chosen_actions": chosen,
                "plan": plan,
            }
        )
        if args.progress_every > 0 and ((idx + 1) % args.progress_every == 0):
            save_planner_progress(args, rows, action_counts, args.checkpoint, device, complete=False)
            elapsed = time.time() - started
            print("[progress] planner saved {}/{} delta_vs_noop_pp={:.3f} paper_delta_pp={:.3f} elapsed_min={:.1f}".format(
                len(rows), len(files),
                mean([row["planned_delta_vs_noop_pp"] for row in rows]),
                mean([(row["planned_home_possession"] - row["real_home_possession"]) * 100.0 for row in rows]),
                elapsed / 60.0,
            ), flush=True)

    summary = build_planner_summary(args, rows, action_counts, args.checkpoint, device, complete=True)
    save_json({"summary": summary, "matches": rows}, args.out_json)
    print(f"saved planner intervention evaluation to {args.out_json}")
    print(summary)


if __name__ == "__main__":
    main()
