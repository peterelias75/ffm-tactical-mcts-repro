#!/usr/bin/env python3
"""Replay saved MCTS decisions and produce Figure-7 match-context evidence.

This does not rerun search. It deterministically replays each saved selected
action with the original advancement seeds, records simulated possession before
every decision, and validates the replayed final possession against the source
result. Panel (a) uses coach possession minus real second-half possession, as
specified by the manuscript; the older E09 aggregation used coach minus no-op.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

from impactful_possession_pipeline.config import DecodeConfig, TokenConfig
from impactful_possession_pipeline.dataset import frames_after_cut, token_index_at_cut
from impactful_possession_pipeline.decode import decode_frames, generate_continuation
from impactful_possession_pipeline.model import load_checkpoint
from impactful_possession_pipeline.statsbomb_features import (
    StructuredStatsBombEncoder,
    get_home_team_id,
    parse_formation,
)
from impactful_possession_pipeline.utils import home_possession_ratio, load_json, mean, save_json
from impactful_possession_pipeline.vocabulary import Vocab


POSSESSION_BINS = (("low", 0.0, 0.45), ("balanced", 0.45, 0.55), ("high", 0.55, 1.000001))


def possession_bin(value: float) -> str:
    for name, low, high in POSSESSION_BINS:
        if low <= value < high:
            return name
    return "high"


def formation_balance(formation) -> int:
    defenders, _, attackers = parse_formation(formation)
    return attackers - defenders


def home_formation_at_cut(events: list, home_team_id, cut_minute: int):
    formation = None
    for event in events:
        if event.get("minute", 0) >= cut_minute:
            continue
        if event.get("team", {}).get("id") != home_team_id:
            continue
        candidate = event.get("tactics", {}).get("formation")
        if candidate not in (None, ""):
            formation = candidate
    return formation


def intervention_intent(action: str, current_formation):
    if action.endswith("NOOP"):
        return "no_action", current_formation
    if action.endswith("_ATTACK"):
        return "attacking", current_formation
    if action.endswith("_DEFEND"):
        return "defending", current_formation
    if action.endswith("_NEUTRAL"):
        return "neutral", current_formation
    if "_SUB_ROLE_" in action:
        role = action.rsplit("_SUB_ROLE_", 1)[1]
        if role in {"ATTACKING_MIDFIELDER", "WINGER", "FORWARD"}:
            return "attacking", current_formation
        if role in {"GOALKEEPER", "DEFENDER", "DEFENSIVE_MIDFIELDER"}:
            return "defending", current_formation
        return "neutral", current_formation
    if "_TACTIC_FORM_" in action:
        new_formation = action.rsplit("_TACTIC_FORM_", 1)[1]
        if current_formation is None:
            return "neutral", new_formation
        old_balance = formation_balance(current_formation)
        new_balance = formation_balance(new_formation)
        intent = "attacking" if new_balance > old_balance else "defending" if new_balance < old_balance else "neutral"
        return intent, new_formation
    return "neutral", current_formation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", required=True)
    parser.add_argument("--lineups-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    source = load_json(args.source_results)
    indexed_source_rows = [
        (index, row) for index, row in enumerate(source["matches"])
        if index % args.shard_count == args.shard_index
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.84, device=0)
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    decode_cfg = DecodeConfig(temperature=source["summary"]["temperature"], top_p=source["summary"]["top_p"])

    out_path = Path(args.out_json)
    replay_rows = []
    if args.resume and out_path.exists():
        replay_rows = load_json(out_path).get("replays", [])
    done = {str(row["match"]) for row in replay_rows}

    def summarize(complete: bool):
        context_groups = defaultdict(list)
        mix = defaultdict(Counter)
        validation_errors = []
        for row in replay_rows:
            context_groups[possession_bin(row["real_home_possession"])].append(row["coach_vs_real_pp"])
            validation_errors.append(abs(row["replayed_mcts_home_possession"] - row["saved_mcts_home_possession"]))
            for decision in row["decisions"]:
                mix[possession_bin(decision["simulated_possession_before_decision"])][decision["intent"]] += 1
        panel_a = {
            name: {"n_matches": len(context_groups[name]), "mean_coach_vs_real_pp": mean(context_groups[name])}
            for name, _, _ in POSSESSION_BINS
        }
        panel_b = {}
        for name, _, _ in POSSESSION_BINS:
            counts = mix[name]
            total = sum(counts.values())
            panel_b[name] = {
                "n_decisions": total,
                "counts": dict(counts),
                "shares": {key: value / total if total else 0.0 for key, value in counts.items()},
            }
        return {
            "experiment": "Figure7_match_context_adapted_plain_mcts",
            "source_results": args.source_results,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "metric_definition_panel_a": "simulated coach possession minus real second-half possession",
            "metric_definition_panel_b": "intervention intent by cumulative simulated second-half possession before each decision",
            "classification_note": "Rich substitution roles use role groups; formation intent is the change in attackers-minus-defenders versus the preceding formation.",
            "num_matches": len(replay_rows),
            "complete": complete,
            "max_abs_replay_error": max(validation_errors) if validation_errors else None,
            "panel_a": panel_a,
            "panel_b": panel_b,
            "replays": replay_rows,
        }

    for index, source_row in indexed_source_rows:
        match_id = str(source_row["match"])
        if match_id in done:
            continue
        event_file = str(Path(args.events_dir) / match_id)
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, args.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        cut_index = token_index_at_cut(frames, args.cut_minute)
        ctx = list(ids[:cut_index])
        current_formation = home_formation_at_cut(events, get_home_team_id(events), args.cut_minute)
        decisions = []
        base_seed = args.seed + index
        saved_by_minute = {item["minute"]: item for item in source_row["interventions"]}
        for minute in range(args.cut_minute, 90):
            second_half_so_far = [
                frame for frame in decode_frames(vocab, ctx, start_minute=0)
                if args.cut_minute <= frame.get("minute", 0) < minute
            ]
            simulated_possession = home_possession_ratio(second_half_so_far)
            action = saved_by_minute[minute]["chosen_action"]
            intent, current_formation = intervention_intent(action, current_formation)
            decisions.append({
                "minute": minute,
                "simulated_possession_before_decision": simulated_possession,
                "chosen_action": action,
                "intent": intent,
            })
            if action != "ACTION::NOOP":
                ctx.append(vocab.id(action))
            generated = generate_continuation(
                model, vocab, ctx, minute, minute + 1,
                token_cfg=token_cfg, decode_cfg=decode_cfg, device=device,
                seed=base_seed + minute + 500_000,
            )
            eos_id = vocab.id("<eos>")
            ctx.extend(token for token in generated if token != eos_id)
        replay_frames = decode_frames(vocab, ctx, start_minute=0)
        replayed_possession = home_possession_ratio(frames_after_cut(replay_frames, args.cut_minute))
        replay_rows.append({
            "match": match_id,
            "real_home_possession": source_row["real_home_possession"],
            "saved_mcts_home_possession": source_row["mcts_home_possession"],
            "replayed_mcts_home_possession": replayed_possession,
            "coach_vs_real_pp": 100.0 * (replayed_possession - source_row["real_home_possession"]),
            "decisions": decisions,
        })
        done.add(match_id)
        save_json(summarize(False), args.out_json)
        print(f"{len(replay_rows)}/{len(indexed_source_rows)} {match_id} replay_error={abs(replayed_possession-source_row['mcts_home_possession']):.8f}", flush=True)

    save_json(summarize(True), args.out_json)
    print(f"saved {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
