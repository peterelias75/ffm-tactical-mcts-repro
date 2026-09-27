import argparse
from collections import Counter
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, TokenConfig
from .dataset import frames_after_cut, token_index_at_cut
from .decode import decode_frames, generate_continuation
from .evaluate_rollout_diagnostics import normalized_l1
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import POSSESSION_EVENT_TYPES, home_possession_ratio, load_json, mean, save_json
from .vocabulary import Vocab


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalized(counter: Counter) -> dict:
    total = sum(counter.values()) or 1
    return {str(k): float(v) / total for k, v in counter.items()}


def sorted_deltas(real: Counter, sim: Counter, limit: int) -> list[dict]:
    real_p = normalized(real)
    sim_p = normalized(sim)
    rows = []
    for key in sorted(set(real_p) | set(sim_p)):
        rows.append({
            "key": key,
            "real_rate": real_p.get(key, 0.0),
            "sim_rate": sim_p.get(key, 0.0),
            "delta_sim_minus_real": sim_p.get(key, 0.0) - real_p.get(key, 0.0),
            "real_count": int(real.get(key, 0)),
            "sim_count": int(sim.get(key, 0)),
        })
    rows.sort(key=lambda row: abs(row["delta_sim_minus_real"]), reverse=True)
    return rows[:limit]


def count_key(frames, key: str) -> Counter:
    return Counter(str(frame.get(key, "unknown")) for frame in frames if frame.get("event_type") != "Action")


def count_locations(frames) -> Counter:
    out = Counter()
    for frame in frames:
        if frame.get("event_type") == "Action":
            continue
        if "lx" in frame and "ly" in frame:
            out[f"{frame['lx']}:{frame['ly']}"] += 1
    return out


def count_transitions(frames) -> Counter:
    out = Counter()
    prev = None
    for frame in frames:
        event_type = frame.get("event_type")
        if event_type == "Action" or event_type is None:
            continue
        if prev is not None:
            out[f"{prev}->{event_type}"] += 1
        prev = event_type
    return out


def count_possession_switch_events(frames) -> Counter:
    out = Counter()
    prev_team = None
    for frame in frames:
        event_type = frame.get("event_type")
        team = frame.get("team")
        if event_type not in POSSESSION_EVENT_TYPES or team not in {"home", "away"}:
            continue
        if prev_team is not None and team != prev_team:
            out[str(event_type)] += 1
        prev_team = team
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze real-vs-sim rollout drift by event type, transition, role, location, and possession switches.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=512)
    parser.add_argument("--possession-guidance-strength", type=float, default=12.0)
    parser.add_argument("--density-from-prefix", action="store_true")
    parser.add_argument("--density-lookback-minutes", type=int, default=15)
    parser.add_argument("--density-multiplier", type=float, default=1.0)
    parser.add_argument("--event-density-tolerance", type=int, default=DecodeConfig.event_density_tolerance)
    parser.add_argument("--prefix-possession-target", action="store_true")
    parser.add_argument("--possession-lookback-minutes", type=int, default=15)
    parser.add_argument("--external-possession-weight", type=float, default=0.75)
    parser.add_argument("--enforce-possession-state", action="store_true")
    parser.add_argument("--football-structure-constraints", action="store_true")
    parser.add_argument("--pass-receipt-boost", type=float, default=0.0)
    parser.add_argument("--pass-carry-boost", type=float, default=0.0)
    parser.add_argument("--overproduced-event-penalty", type=float, default=0.0)
    parser.add_argument("--max-pass-run", type=int, default=DecodeConfig.max_pass_run)
    parser.add_argument("--force-receipt-after-pass-run", type=int, default=DecodeConfig.force_receipt_after_pass_run)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    if "test_files" not in extra or "data_cfg" not in extra:
        raise RuntimeError("Checkpoint does not include test_files/data_cfg metadata.")

    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig()
    data_cfg = DataConfig(**extra["data_cfg"])
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)

    real_event = Counter(); sim_event = Counter()
    real_team = Counter(); sim_team = Counter()
    real_role = Counter(); sim_role = Counter()
    real_loc = Counter(); sim_loc = Counter()
    real_trans = Counter(); sim_trans = Counter()
    real_switch = Counter(); sim_switch = Counter()
    rows = []

    for idx, event_file in enumerate(list(extra["test_files"])[: args.max_files]):
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, data_cfg.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        cut_idx = token_index_at_cut(frames, args.cut_minute)
        real_frames = frames_after_cut(frames, args.cut_minute, args.end_minute - args.cut_minute)

        target_events_per_minute = None
        if args.density_from_prefix:
            start = max(0, args.cut_minute - args.density_lookback_minutes)
            prefix_frames = [frame for frame in frames if start <= frame.get("minute", 0) < args.cut_minute]
            target_events_per_minute = len(prefix_frames) / max(1, args.cut_minute - start)
            target_events_per_minute *= max(0.1, args.density_multiplier)
        target_home_possession = None
        if args.prefix_possession_target:
            start = max(0, args.cut_minute - args.possession_lookback_minutes)
            prefix_frames = [frame for frame in frames if start <= frame.get("minute", 0) < args.cut_minute]
            target_home_possession = home_possession_ratio(prefix_frames)

        decode_cfg = DecodeConfig(
            top_p=args.top_p,
            temperature=args.temperature,
            prompt_tail_tokens=args.prompt_tail_tokens,
            target_events_per_minute=target_events_per_minute,
            event_density_tolerance=args.event_density_tolerance,
            target_home_possession=target_home_possession,
            external_possession_weight=args.external_possession_weight,
            possession_guidance_strength=args.possession_guidance_strength,
            enforce_possession_state=args.enforce_possession_state,
            football_structure_constraints=args.football_structure_constraints,
            pass_receipt_boost=args.pass_receipt_boost,
            pass_carry_boost=args.pass_carry_boost,
            overproduced_event_penalty=args.overproduced_event_penalty,
            max_pass_run=args.max_pass_run,
            force_receipt_after_pass_run=args.force_receipt_after_pass_run,
        )
        generated = generate_continuation(model, vocab, ids[:cut_idx], args.cut_minute, args.end_minute, token_cfg, decode_cfg, device, args.seed + idx)
        sim_frames = decode_frames(vocab, generated, start_minute=args.cut_minute)

        real_event.update(count_key(real_frames, "event_type")); sim_event.update(count_key(sim_frames, "event_type"))
        real_team.update(count_key(real_frames, "team")); sim_team.update(count_key(sim_frames, "team"))
        real_role.update(count_key(real_frames, "role")); sim_role.update(count_key(sim_frames, "role"))
        real_loc.update(count_locations(real_frames)); sim_loc.update(count_locations(sim_frames))
        real_trans.update(count_transitions(real_frames)); sim_trans.update(count_transitions(sim_frames))
        real_switch.update(count_possession_switch_events(real_frames)); sim_switch.update(count_possession_switch_events(sim_frames))
        rows.append({
            "match": Path(event_file).name,
            "real_frames": len(real_frames),
            "sim_frames": len(sim_frames),
            "real_home_possession": home_possession_ratio(real_frames),
            "sim_home_possession": home_possession_ratio(sim_frames),
        })

    summary = {
        "checkpoint": args.checkpoint,
        "num_matches": len(rows),
        "cut_minute": args.cut_minute,
        "end_minute": args.end_minute,
        "mean_real_frames": mean([r["real_frames"] for r in rows]),
        "mean_sim_frames": mean([r["sim_frames"] for r in rows]),
        "mean_abs_possession_error_pp": mean([abs(r["sim_home_possession"] - r["real_home_possession"]) * 100.0 for r in rows]),
        "event_type_l1": normalized_l1(real_event, sim_event),
        "team_l1": normalized_l1(real_team, sim_team),
        "role_l1": normalized_l1(real_role, sim_role),
        "location_l1": normalized_l1(real_loc, sim_loc),
        "transition_l1": normalized_l1(real_trans, sim_trans),
    }
    output = {
        "summary": summary,
        "top_event_type_drift": sorted_deltas(real_event, sim_event, args.limit),
        "top_team_drift": sorted_deltas(real_team, sim_team, args.limit),
        "top_role_drift": sorted_deltas(real_role, sim_role, args.limit),
        "top_location_drift": sorted_deltas(real_loc, sim_loc, args.limit),
        "top_transition_drift": sorted_deltas(real_trans, sim_trans, args.limit),
        "top_possession_switch_event_drift": sorted_deltas(real_switch, sim_switch, args.limit),
        "raw_counts": {
            "real_event_type": dict(real_event.most_common()),
            "sim_event_type": dict(sim_event.most_common()),
            "real_transition": dict(real_trans.most_common(200)),
            "sim_transition": dict(sim_trans.most_common(200)),
        },
        "matches": rows,
    }
    save_json(output, args.out_json)
    print(f"saved rollout drift analysis to {args.out_json}")
    print(summary)
    print("top_event_type_drift", output["top_event_type_drift"][:8])


if __name__ == "__main__":
    main()
