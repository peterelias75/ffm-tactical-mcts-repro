import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from .config import DataConfig, DecodeConfig, TokenConfig
from .dataset import frames_after_cut, token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import POSSESSION_EVENT_TYPES, home_possession_ratio, load_json, mean, save_json
from .vocabulary import Vocab


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalized_l1(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values()) or 1
    right_total = sum(right.values()) or 1
    return float(sum(abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total) for key in keys))


def count_values(frames: Iterable[dict], key: str) -> Counter:
    return Counter(frame.get(key, "unknown") for frame in frames)


def count_locations(frames: Iterable[dict]) -> Counter:
    counts = Counter()
    for frame in frames:
        if "lx" in frame and "ly" in frame:
            counts[f"{frame['lx']}:{frame['ly']}"] += 1
    return counts



LEGAL_POSSESSION_SWITCH_EVENT_TYPES = {
    "Ball Recovery",
    "Interception",
    "Duel",
    "Clearance",
    "Miscontrol",
    "Dispossessed",
    "Foul Won",
    "Foul Committed",
    "Goal Keeper",
    "Block",
    "Error",
}


def possession_transition_metrics(frames: Iterable[dict]) -> dict:
    previous_team = None
    possession_events = 0
    switches = 0
    illegal_switches = 0
    for frame in frames:
        event_type = frame.get("event_type")
        team = frame.get("team")
        if event_type not in POSSESSION_EVENT_TYPES or team not in {"home", "away"}:
            continue
        possession_events += 1
        if previous_team is not None and team != previous_team:
            switches += 1
            if event_type not in LEGAL_POSSESSION_SWITCH_EVENT_TYPES:
                illegal_switches += 1
        previous_team = team
    return {
        "possession_event_count": possession_events,
        "possession_switch_count": switches,
        "illegal_possession_switch_count": illegal_switches,
        "illegal_possession_switch_rate": illegal_switches / switches if switches else 0.0,
    }


def frames_in_window(frames: Iterable[dict], cut_minute: int, horizon_minutes: int) -> list[dict]:
    end_minute = cut_minute + horizon_minutes
    return [frame for frame in frames if cut_minute <= frame.get("minute", cut_minute) < end_minute]


def parse_horizons(text: str, max_horizon: int) -> list[int]:
    horizons = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        horizon = int(piece)
        if 0 < horizon <= max_horizon and horizon not in horizons:
            horizons.append(horizon)
    return horizons

def summarize_match(real_frames: list[dict], sim_frames: list[dict], cut_minute: int, end_minute: int) -> dict:
    horizon = max(1, end_minute - cut_minute)
    real_pos = home_possession_ratio(real_frames)
    sim_pos = home_possession_ratio(sim_frames)
    real_transition = possession_transition_metrics(real_frames)
    sim_transition = possession_transition_metrics(sim_frames)
    return {
        "real_home_possession": real_pos,
        "sim_home_possession": sim_pos,
        "abs_possession_error_pp": abs(sim_pos - real_pos) * 100.0,
        "real_frame_count": len(real_frames),
        "sim_frame_count": len(sim_frames),
        "real_events_per_minute": len(real_frames) / horizon,
        "sim_events_per_minute": len(sim_frames) / horizon,
        "event_type_l1": normalized_l1(count_values(real_frames, "event_type"), count_values(sim_frames, "event_type")),
        "team_l1": normalized_l1(count_values(real_frames, "team"), count_values(sim_frames, "team")),
        "location_l1": normalized_l1(count_locations(real_frames), count_locations(sim_frames)),
        "real_possession_switch_count": real_transition["possession_switch_count"],
        "sim_possession_switch_count": sim_transition["possession_switch_count"],
        "real_illegal_possession_switch_rate": real_transition["illegal_possession_switch_rate"],
        "sim_illegal_possession_switch_rate": sim_transition["illegal_possession_switch_rate"],
        "real_illegal_possession_switch_count": real_transition["illegal_possession_switch_count"],
        "sim_illegal_possession_switch_count": sim_transition["illegal_possession_switch_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded rollout diagnostics on the exact checkpoint holdout split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/rollout_diagnostics.json")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=512)
    parser.add_argument("--min-events-per-minute", type=int, default=DecodeConfig.min_events_per_minute)
    parser.add_argument("--max-events-per-minute", type=int, default=DecodeConfig.max_events_per_minute)
    parser.add_argument("--possession-guidance-strength", type=float, default=DecodeConfig.possession_guidance_strength)
    parser.add_argument("--target-events-per-minute", type=float, default=None)
    parser.add_argument("--density-from-prefix", action="store_true", help="Use the pre-cut match event density as the rollout pacing target.")
    parser.add_argument("--density-lookback-minutes", type=int, default=15)
    parser.add_argument("--density-multiplier", type=float, default=1.0, help="Multiply the prefix/explicit event-density target before decoding.")
    parser.add_argument("--event-density-tolerance", type=int, default=DecodeConfig.event_density_tolerance)
    parser.add_argument("--prefix-possession-target", action="store_true", help="Blend pre-cut home possession into the decoder team target.")
    parser.add_argument("--possession-lookback-minutes", type=int, default=15)
    parser.add_argument("--external-possession-weight", type=float, default=0.0)
    parser.add_argument("--possession-target-calibration-json", default=None, help="Optional JSON with intercept, prefix_coef, and optional full_prefix_coef for calibrated possession targets.")
    parser.add_argument("--possession-guidance-prior-events", type=float, default=0.0, help="Pseudo-count possession events initialized from the decoder target.")
    parser.add_argument("--enforce-possession-state", action="store_true")
    parser.add_argument("--prior-json", default=None, help="Optional rollout prior JSON from calibrate_rollout_priors.")
    parser.add_argument("--prior-bias-strength", type=float, default=0.0)
    parser.add_argument("--conditional-prior-json", default=None, help="Optional conditional rollout prior JSON.")
    parser.add_argument("--conditional-prior-bias-strength", type=float, default=0.0)
    parser.add_argument("--football-structure-constraints", action="store_true")
    parser.add_argument("--pass-receipt-boost", type=float, default=0.0)
    parser.add_argument("--pass-carry-boost", type=float, default=0.0)
    parser.add_argument("--overproduced-event-penalty", type=float, default=0.0)
    parser.add_argument("--max-pass-run", type=int, default=DecodeConfig.max_pass_run)
    parser.add_argument("--force-receipt-after-pass-run", type=int, default=DecodeConfig.force_receipt_after_pass_run)
    parser.add_argument("--drift-horizons", default="1,3,5,10,15")
    parser.add_argument("--no-organic-actions", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    if "test_files" not in extra or "data_cfg" not in extra:
        raise RuntimeError("Checkpoint does not include test_files/data_cfg metadata.")

    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig()
    data_cfg = DataConfig(**extra["data_cfg"])
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)

    horizons = parse_horizons(args.drift_horizons, args.end_minute - args.cut_minute)
    prior_cfg = load_json(args.prior_json) if args.prior_json else {}
    conditional_prior_cfg = load_json(args.conditional_prior_json) if args.conditional_prior_json else {}
    possession_target_calibration = load_json(args.possession_target_calibration_json) if args.possession_target_calibration_json else {}

    rows = []
    for idx, event_file in enumerate(list(extra["test_files"])[: args.max_files]):
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, data_cfg.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        cut_idx = token_index_at_cut(frames, args.cut_minute)
        real_frames = frames_after_cut(frames, args.cut_minute, args.end_minute - args.cut_minute)
        target_events_per_minute = args.target_events_per_minute
        density_prefix_frames = None
        if args.density_from_prefix:
            lookback_start = max(0, args.cut_minute - args.density_lookback_minutes)
            density_prefix_frames = [frame for frame in frames if lookback_start <= frame.get("minute", 0) < args.cut_minute]
            target_events_per_minute = len(density_prefix_frames) / max(1, args.cut_minute - lookback_start)
        if target_events_per_minute is not None:
            target_events_per_minute *= max(0.1, float(args.density_multiplier))
        target_home_possession = None
        if args.prefix_possession_target:
            possession_lookback_start = max(0, args.cut_minute - args.possession_lookback_minutes)
            possession_prefix_frames = [frame for frame in frames if possession_lookback_start <= frame.get("minute", 0) < args.cut_minute]
            target_home_possession = home_possession_ratio(possession_prefix_frames)
            if possession_target_calibration:
                full_prefix_frames = [frame for frame in frames if 0 <= frame.get("minute", 0) < args.cut_minute]
                full_prefix_possession = home_possession_ratio(full_prefix_frames)
                target_home_possession = (
                    float(possession_target_calibration.get("intercept", 0.0))
                    + float(possession_target_calibration.get("prefix_coef", 1.0)) * float(target_home_possession)
                    + float(possession_target_calibration.get("full_prefix_coef", 0.0)) * float(full_prefix_possession)
                )
                target_home_possession = max(0.05, min(0.95, target_home_possession))
        decode_cfg = DecodeConfig(
            top_p=args.top_p,
            temperature=args.temperature,
            prompt_tail_tokens=args.prompt_tail_tokens,
            min_events_per_minute=args.min_events_per_minute,
            max_events_per_minute=args.max_events_per_minute,
            target_events_per_minute=target_events_per_minute,
            event_density_tolerance=args.event_density_tolerance,
            target_home_possession=target_home_possession,
            external_possession_weight=args.external_possession_weight,
            possession_guidance_prior_events=args.possession_guidance_prior_events,
            include_organic_actions=not args.no_organic_actions,
            possession_guidance_strength=args.possession_guidance_strength,
            enforce_possession_state=args.enforce_possession_state,
            event_type_prior=prior_cfg.get("event_type"),
            role_prior=prior_cfg.get("role"),
            lx_prior=prior_cfg.get("lx"),
            ly_prior=prior_cfg.get("ly"),
            prior_bias_strength=args.prior_bias_strength,
            conditional_prior=conditional_prior_cfg,
            conditional_prior_bias_strength=args.conditional_prior_bias_strength,
            football_structure_constraints=args.football_structure_constraints,
            pass_receipt_boost=args.pass_receipt_boost,
            pass_carry_boost=args.pass_carry_boost,
            overproduced_event_penalty=args.overproduced_event_penalty,
            max_pass_run=args.max_pass_run,
            force_receipt_after_pass_run=args.force_receipt_after_pass_run,
        )
        generated = generate_continuation(
            model,
            vocab,
            ids[:cut_idx],
            start_minute=args.cut_minute,
            end_minute=args.end_minute,
            token_cfg=token_cfg,
            decode_cfg=decode_cfg,
            device=device,
            seed=args.seed + idx,
        )
        sim_frames = decode_frames(vocab, generated, start_minute=args.cut_minute)
        full_summary = summarize_match(real_frames, sim_frames, args.cut_minute, args.end_minute)
        drift_by_horizon = {
            f"{horizon}m": summarize_match(
                frames_in_window(real_frames, args.cut_minute, horizon),
                frames_in_window(sim_frames, args.cut_minute, horizon),
                args.cut_minute,
                args.cut_minute + horizon,
            )
            for horizon in horizons
        }
        rows.append(
            {
                "match": Path(event_file).name,
                "target_events_per_minute": target_events_per_minute,
                "target_home_possession": target_home_possession,
                **full_summary,
                "drift_by_horizon": drift_by_horizon,
            }
        )
        partial_summary = {
            "checkpoint": args.checkpoint,
            "device": str(device),
            "complete": False,
            "num_matches": len(rows),
            "requested_max_files": args.max_files,
            "cut_minute": args.cut_minute,
            "end_minute": args.end_minute,
            "mean_abs_possession_error_pp": mean([row["abs_possession_error_pp"] for row in rows]),
            "mean_real_events_per_minute": mean([row["real_events_per_minute"] for row in rows]),
            "mean_sim_events_per_minute": mean([row["sim_events_per_minute"] for row in rows]),
            "mean_event_type_l1": mean([row["event_type_l1"] for row in rows]),
            "mean_team_l1": mean([row["team_l1"] for row in rows]),
            "mean_location_l1": mean([row["location_l1"] for row in rows]),
            "mean_real_illegal_possession_switch_rate": mean([row["real_illegal_possession_switch_rate"] for row in rows]),
            "mean_sim_illegal_possession_switch_rate": mean([row["sim_illegal_possession_switch_rate"] for row in rows]),
            "possession_target_calibration_json": args.possession_target_calibration_json,
            "possession_guidance_prior_events": args.possession_guidance_prior_events,
        }
        save_json({"summary": partial_summary, "matches": rows}, args.out_json)
        print("[progress] {}/{} mean_abs_possession_error_pp={:.3f}".format(idx + 1, args.max_files, partial_summary["mean_abs_possession_error_pp"]), flush=True)

    drift_summary = {}
    for horizon in horizons:
        key = f"{horizon}m"
        drift_summary[key] = {
            "mean_abs_possession_error_pp": mean([row["drift_by_horizon"][key]["abs_possession_error_pp"] for row in rows]),
            "mean_team_l1": mean([row["drift_by_horizon"][key]["team_l1"] for row in rows]),
            "mean_sim_illegal_possession_switch_rate": mean([row["drift_by_horizon"][key]["sim_illegal_possession_switch_rate"] for row in rows]),
            "mean_sim_events_per_minute": mean([row["drift_by_horizon"][key]["sim_events_per_minute"] for row in rows]),
            "mean_real_events_per_minute": mean([row["drift_by_horizon"][key]["real_events_per_minute"] for row in rows]),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "complete": True,
        "requested_max_files": args.max_files,
        "num_matches": len(rows),
        "cut_minute": args.cut_minute,
        "end_minute": args.end_minute,
        "mean_abs_possession_error_pp": mean([row["abs_possession_error_pp"] for row in rows]),
        "mean_real_events_per_minute": mean([row["real_events_per_minute"] for row in rows]),
        "mean_sim_events_per_minute": mean([row["sim_events_per_minute"] for row in rows]),
        "mean_event_type_l1": mean([row["event_type_l1"] for row in rows]),
        "mean_team_l1": mean([row["team_l1"] for row in rows]),
        "mean_location_l1": mean([row["location_l1"] for row in rows]),
        "mean_real_illegal_possession_switch_rate": mean([row["real_illegal_possession_switch_rate"] for row in rows]),
        "mean_sim_illegal_possession_switch_rate": mean([row["sim_illegal_possession_switch_rate"] for row in rows]),
        "enforce_possession_state": args.enforce_possession_state,
        "prior_json": args.prior_json,
        "prior_bias_strength": args.prior_bias_strength,
        "conditional_prior_json": args.conditional_prior_json,
        "conditional_prior_bias_strength": args.conditional_prior_bias_strength,
        "football_structure_constraints": args.football_structure_constraints,
        "pass_receipt_boost": args.pass_receipt_boost,
        "pass_carry_boost": args.pass_carry_boost,
        "overproduced_event_penalty": args.overproduced_event_penalty,
        "max_pass_run": args.max_pass_run,
        "force_receipt_after_pass_run": args.force_receipt_after_pass_run,
        "possession_target_calibration_json": args.possession_target_calibration_json,
        "possession_guidance_prior_events": args.possession_guidance_prior_events,
        "drift_by_horizon": drift_summary,
    }
    save_json({"summary": summary, "matches": rows}, args.out_json)
    print(f"saved rollout diagnostics to {args.out_json}")
    print(summary)


if __name__ == "__main__":
    main()
