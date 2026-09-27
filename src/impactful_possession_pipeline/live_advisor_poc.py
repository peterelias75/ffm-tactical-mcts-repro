"""Proof-of-concept: run the existing forward-simulator + planner as a live,
minute-by-minute in-match advisor instead of an offline plan_second_half()
call over a whole finished file.

This does NOT add a new data source. It takes one real (held-out) StatsBomb
match and replays it minute by minute, at each step only exposing the model
to the token prefix that would actually be available up to that point in a
live feed -- exactly the constraint a real live integration would have. At
each decision tick it calls the same cheap, deterministic planner.choose_action
value-head path already used by plan_second_half(), and prints a live
recommendation feed plus per-decision latency, which is the number that
actually matters for "can this run during a real match."

Usage:
    python3 -m impactful_possession_pipeline.live_advisor_poc \
        --checkpoint impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt \
        --match-json /mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/events/19756.json
"""

import argparse
import time
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, PlannerConfig, TokenConfig
from .dataset import token_index_at_cut
from .model import load_checkpoint
from .planner import action_kind, choose_action
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, load_json
from .vocabulary import Vocab


def format_score(score) -> str:
    return f"{score.action:<32} utility={score.utility:+.4f} delta={score.delta * 100:+.2f}pp unc={score.uncertainty:.3f}"


def real_interventions_in_window(frames, start_minute: int, end_minute: int):
    return [
        f for f in frames
        if f.get("event_type") == "Action" and start_minute <= f.get("minute", 0) < end_minute
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live, minute-by-minute replay of the planner as an in-match advisor (POC).")
    parser.add_argument("--checkpoint", default="impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt")
    parser.add_argument("--match-json", default="/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/events/19756.json")
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--start-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=90)
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index to pin to (defaults to first visible).")
    parser.add_argument("--require-confirmation", action="store_true", help="Also run the slow generation-based confirmation pass on shortlisted candidates.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if (args.gpu is not None and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"loading checkpoint {args.checkpoint} on {device} ...", flush=True)
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg_dict = extra.get("token_cfg") or {}
    token_cfg = TokenConfig(**token_cfg_dict) if token_cfg_dict else TokenConfig()
    print(f"checkpoint epoch={extra.get('epoch')} val_pos_mae_pp={extra.get('val_pos_mae_pp')} rich_actions={token_cfg.rich_actions}", flush=True)

    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    events = load_json(args.match_json)
    lineup_path = encoder.lineup_path_for_event_file(args.match_json, args.lineups_dir)
    ids, frames = encoder.encode_match(events, lineup_path)
    print(f"match={Path(args.match_json).name} total_frames={len(frames)} total_tokens={len(ids)}", flush=True)

    planner_cfg = PlannerConfig()
    decode_cfg = DecodeConfig()

    used_subs = 0
    used_tactics = 0
    cooldown = 0
    accepted = []

    print("\n=== LIVE ADVISOR FEED (replaying real match events minute-by-minute) ===\n", flush=True)
    for minute in range(args.start_minute, args.end_minute, planner_cfg.step_minutes):
        prefix_idx = token_index_at_cut(frames, minute)
        prompt_ids = ids[:prefix_idx]

        t0 = time.perf_counter()
        action, scores, confirmations = choose_action(
            model, vocab, prompt_ids, minute, used_subs, used_tactics, cooldown,
            token_cfg, decode_cfg, planner_cfg, device, seed=args.seed + minute,
            require_confirmation=args.require_confirmation,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"[minute {minute:>3}] tokens_seen={len(prompt_ids):>5} decision_latency={latency_ms:6.1f}ms", flush=True)
        for score in sorted(scores, key=lambda s: s.utility, reverse=True)[:4]:
            marker = " <-- CHOSEN" if score.action == action else ""
            print(f"    {format_score(score)}{marker}", flush=True)
        for confirmation in confirmations:
            print(f"    confirm[{confirmation.action}] seeds={confirmation.seeds} mean_delta={confirmation.mean_delta*100:+.2f}pp lcb={confirmation.lcb*100:+.2f}pp pass={confirmation.passes}", flush=True)

        if action != "ACTION::NOOP":
            accepted.append((minute, action))
            if action_kind(action) == "sub":
                used_subs += 1
            elif action_kind(action) == "tactic":
                used_tactics += 1
                cooldown = planner_cfg.tactical_cooldown_steps
        else:
            cooldown = max(0, cooldown - 1)
        print("", flush=True)

    print("=== SUMMARY ===", flush=True)
    print(f"advisor recommended {len(accepted)} intervention(s): {accepted}", flush=True)

    real_actions = real_interventions_in_window(frames, args.start_minute, args.end_minute)
    print(f"real match actually had {len(real_actions)} sub/tactical event(s) in this window: "
          f"{[(a['minute'], a['action']) for a in real_actions]}", flush=True)

    full_prefix_idx = token_index_at_cut(frames, args.start_minute)
    real_rest = [f for f in frames if f.get("minute", 0) >= args.start_minute]
    print(f"real observed home possession from minute {args.start_minute} onward: "
          f"{home_possession_ratio(real_rest) * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
