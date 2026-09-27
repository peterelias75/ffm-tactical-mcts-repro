import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, load_json, mean, save_json, std
from .vocabulary import Vocab


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_decode_cfg(args, frames: list[dict], cut_minute: int, conditional_prior_cfg: dict | None = None) -> DecodeConfig:
    target_events_per_minute = args.target_events_per_minute
    if args.density_from_prefix:
        start = max(0, cut_minute - args.density_lookback_minutes)
        prefix = [frame for frame in frames if start <= frame.get("minute", 0) < cut_minute]
        target_events_per_minute = len(prefix) / max(1, cut_minute - start)
    if target_events_per_minute is not None:
        target_events_per_minute *= max(0.1, float(args.density_multiplier))

    target_home_possession = None
    if args.prefix_possession_target:
        start = max(0, cut_minute - args.possession_lookback_minutes)
        prefix = [frame for frame in frames if start <= frame.get("minute", 0) < cut_minute]
        target_home_possession = home_possession_ratio(prefix)

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


def build_summary(args, rows: list[dict], action_counts: Counter, total_candidates: int, checkpoint: str, device: torch.device, complete: bool) -> dict:
    return {
        "checkpoint": checkpoint,
        "value_head_json": args.value_head_json,
        "device": str(device),
        "offset": args.offset,
        "num_matches": len(rows),
        "total_filtered_candidates": total_candidates,
        "min_value_head_delta": args.min_value_head_delta,
        "exclude_actions": sorted({action.strip() for action in args.exclude_actions.split(",") if action.strip()}),
        "only_actions": sorted({action.strip() for action in args.only_actions.split(",") if action.strip()}),
        "score_action": args.score_action,
        "cut_minute": args.cut_minute,
        "end_minute": args.end_minute,
        "rollout_seeds": args.rollout_seeds,
        "conditional_prior_json": args.conditional_prior_json,
        "conditional_prior_bias_strength": args.conditional_prior_bias_strength,
        "complete": complete,
        "updated_at_unix": time.time(),
        "mean_value_head_delta_pp": mean([row["value_head_delta_pp"] for row in rows]),
        "mean_rollout_delta_pp": mean([row["mean_rollout_delta_pp"] for row in rows]),
        "median_positive_seed_rate": sorted([row["positive_seed_rate"] for row in rows])[len(rows) // 2] if rows else 0.0,
        "match_positive_rate": mean([1.0 if row["mean_rollout_delta_pp"] > 0 else 0.0 for row in rows]),
        "chosen_action_counts": dict(action_counts.most_common()),
    }


def save_progress(args, rows: list[dict], action_counts: Counter, total_candidates: int, checkpoint: str, device: torch.device, complete: bool) -> None:
    summary = build_summary(args, rows, action_counts, total_candidates, checkpoint, device, complete)
    save_json({"summary": summary, "matches": rows}, args.out_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired rollout validation for value-head shortlisted interventions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--value-head-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all accepted rows after offset.")
    parser.add_argument("--min-value-head-delta", type=float, default=0.0, help="Minimum accepted value-head delta, in possession fraction units.")
    parser.add_argument("--max-value-head-delta", type=float, default=0.0, help="Maximum accepted value-head delta, in possession fraction units. 0 disables.")
    parser.add_argument("--max-base-uncertainty", type=float, default=0.0, help="Maximum base uncertainty from value-head screen, in fraction units. 0 disables.")
    parser.add_argument("--exclude-actions", default="", help="Comma-separated action tokens to remove before chunking.")
    parser.add_argument("--only-actions", default="", help="Comma-separated action tokens to keep before chunking.")
    parser.add_argument("--score-action", default="", help="Validate this action from each row's score table instead of using accepted chosen actions.")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=60)
    parser.add_argument("--rollout-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=512)
    parser.add_argument("--possession-guidance-strength", type=float, default=12.0)
    parser.add_argument("--target-events-per-minute", type=float, default=None)
    parser.add_argument("--density-from-prefix", action="store_true")
    parser.add_argument("--density-lookback-minutes", type=int, default=15)
    parser.add_argument("--density-multiplier", type=float, default=1.12)
    parser.add_argument("--event-density-tolerance", type=int, default=2)
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

    shortlist = json.loads(Path(args.value_head_json).read_text())
    if args.score_action:
        accepted = []
        for row in shortlist["matches"]:
            score = next((score for score in row.get("scores", []) if score.get("action") == args.score_action), None)
            if score is None:
                continue
            if args.min_value_head_delta > 0 and float(score.get("delta", 0.0)) < args.min_value_head_delta:
                continue
            if args.max_value_head_delta > 0 and float(score.get("delta", 0.0)) > args.max_value_head_delta:
                continue
            if args.max_base_uncertainty > 0 and float(row.get("base_uncertainty", 0.0)) > args.max_base_uncertainty:
                continue
            accepted.append({
                "match": row["match"],
                "chosen_action": args.score_action,
                "best_delta": float(score.get("delta", 0.0)),
                "best_utility": float(score.get("utility", 0.0)),
            })
    else:
        accepted = [row for row in shortlist["matches"] if row.get("accepted")]
        if args.min_value_head_delta > 0:
            accepted = [row for row in accepted if float(row.get("best_delta", 0.0)) >= args.min_value_head_delta]
        if args.max_value_head_delta > 0:
            accepted = [row for row in accepted if float(row.get("best_delta", 0.0)) <= args.max_value_head_delta]
        if args.max_base_uncertainty > 0:
            accepted = [row for row in accepted if float(row.get("base_uncertainty", 0.0)) <= args.max_base_uncertainty]
    excluded = {action.strip() for action in args.exclude_actions.split(",") if action.strip()}
    if excluded:
        accepted = [row for row in accepted if row.get("chosen_action") not in excluded]
    only = {action.strip() for action in args.only_actions.split(",") if action.strip()}
    if only:
        accepted = [row for row in accepted if row.get("chosen_action") in only]
    total_candidates = len(accepted)
    accepted = accepted[args.offset:]
    if args.max_files > 0:
        accepted = accepted[: args.max_files]

    event_by_name = {Path(path).name: path for path in extra["test_files"]}
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    data_cfg = DataConfig(**extra["data_cfg"])
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)

    rows = []
    action_counts = Counter()
    started = time.time()
    for row_idx, item in enumerate(accepted):
        print("[progress] match {}/{} offset={} action={} match={}".format(row_idx + 1, len(accepted), args.offset, item["chosen_action"], item["match"]), flush=True)
        event_file = event_by_name[item["match"]]
        action = item["chosen_action"]
        action_id = vocab.id(action)
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, data_cfg.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        prompt_ids = ids[: token_index_at_cut(frames, args.cut_minute)]
        decode_cfg = build_decode_cfg(args, frames, args.cut_minute, conditional_prior_cfg)

        deltas = []
        noop_values = []
        action_values = []
        for seed_idx in range(args.rollout_seeds):
            seed = args.seed + args.offset * 10000 + row_idx * 100 + seed_idx
            noop_generated = generate_continuation(
                model, vocab, prompt_ids, args.cut_minute, args.end_minute,
                token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=seed,
            )
            action_generated = generate_continuation(
                model, vocab, prompt_ids, args.cut_minute, args.end_minute,
                token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=seed, injected_action_id=action_id,
            )
            noop_pos = home_possession_ratio(decode_frames(vocab, noop_generated, start_minute=args.cut_minute))
            action_pos = home_possession_ratio(decode_frames(vocab, action_generated, start_minute=args.cut_minute))
            noop_values.append(noop_pos)
            action_values.append(action_pos)
            deltas.append((action_pos - noop_pos) * 100.0)

        action_counts[action] += 1
        rows.append(
            {
                "match": item["match"],
                "action": action,
                "value_head_delta_pp": item.get("best_delta", 0.0) * 100.0,
                "mean_noop_home_possession": mean(noop_values),
                "mean_action_home_possession": mean(action_values),
                "mean_rollout_delta_pp": mean(deltas),
                "std_rollout_delta_pp": std(deltas),
                "positive_seed_rate": mean([1.0 if delta > 0 else 0.0 for delta in deltas]),
                "seed_deltas_pp": deltas,
            }
        )
        if args.progress_every > 0 and ((row_idx + 1) % args.progress_every == 0):
            save_progress(args, rows, action_counts, total_candidates, args.checkpoint, device, complete=False)
            elapsed = time.time() - started
            current_mean = mean([row["mean_rollout_delta_pp"] for row in rows])
            print(f"[progress] saved {len(rows)}/{len(accepted)} mean_delta_pp={current_mean:.3f} elapsed_min={elapsed / 60.0:.1f}", flush=True)

    summary = build_summary(args, rows, action_counts, total_candidates, args.checkpoint, device, complete=True)
    save_json({"summary": summary, "matches": rows}, args.out_json)
    print(f"saved shortlisted intervention rollout validation to {args.out_json}")
    print(summary)


if __name__ == "__main__":
    main()
