import argparse
import math
from pathlib import Path

import torch

from .config import DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import POSSESSION_EVENT_TYPES, home_possession_ratio, load_json, mean, save_json, std
from .vocabulary import Vocab


def sem_lcb(values: list[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def summarize(deltas: list[float]) -> dict:
    return {
        "n": len(deltas),
        "mean_delta_pp": mean(deltas),
        "std_delta_pp": std(deltas),
        "lcb90_delta_pp": sem_lcb(deltas),
        "positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in deltas]),
    }


def select_candidates(data: dict, top_k: int) -> list[dict]:
    candidates = data.get("top_candidates") or data.get("candidates") or []
    return candidates[:top_k]


def branch_metrics(frames: list[dict], cut_minute: int, end_minute: int) -> dict:
    horizon = max(1, end_minute - cut_minute)
    possession_frames = [
        frame
        for frame in frames
        if frame.get("event_type") in POSSESSION_EVENT_TYPES and frame.get("team") in {"home", "away"}
    ]
    home_possession = home_possession_ratio(frames)
    home_events = sum(1 for frame in possession_frames if frame.get("team") == "home")
    away_events = sum(1 for frame in possession_frames if frame.get("team") == "away")
    return {
        "home_possession": home_possession,
        "frames": len(frames),
        "possession_frames": len(possession_frames),
        "home_possession_events": home_events,
        "away_possession_events": away_events,
        "events_per_minute": len(frames) / horizon,
        "possession_events_per_minute": len(possession_frames) / horizon,
    }


def sanity_reasons(metrics: dict, args: argparse.Namespace) -> list[str]:
    reasons = []
    possession = float(metrics["home_possession"])
    if possession <= args.reject_possession_below:
        reasons.append("home_possession_too_low")
    if possession >= args.reject_possession_above:
        reasons.append("home_possession_too_high")
    if int(metrics["possession_frames"]) < args.min_possession_frames:
        reasons.append("too_few_possession_frames")
    if float(metrics["events_per_minute"]) < args.min_events_per_minute:
        reasons.append("events_per_minute_too_low")
    if float(metrics["events_per_minute"]) > args.max_events_per_minute:
        reasons.append("events_per_minute_too_high")
    return reasons


def prefix_home_possession_target(frames: list[dict], cut_minute: int, lookback_minutes: int) -> float | None:
    start_minute = max(0, cut_minute - lookback_minutes)
    prefix_frames = [
        frame
        for frame in frames
        if start_minute <= frame.get("minute", 0) < cut_minute
    ]
    if not prefix_frames:
        return None
    return home_possession_ratio(prefix_frames)


def make_decode_cfg(
    args: argparse.Namespace,
    target_home_possession: float | None = None,
    target_events_per_minute: float | None = None,
    conditional_prior: dict | None = None,
) -> DecodeConfig:
    return DecodeConfig(
        top_p=args.top_p,
        temperature=args.temperature,
        prompt_tail_tokens=args.prompt_tail_tokens,
        target_events_per_minute=target_events_per_minute,
        event_density_tolerance=args.event_density_tolerance,
        target_home_possession=target_home_possession,
        external_possession_weight=args.external_possession_weight,
        possession_guidance_prior_events=args.possession_guidance_prior_events,
        include_organic_actions=not args.no_organic_actions,
        conditional_prior=conditional_prior,
        conditional_prior_bias_strength=args.conditional_prior_bias_strength,
    )


def replay_with_metrics(
    model,
    vocab: Vocab,
    prompt_ids: list[int],
    actions: list[str],
    cut_minute: int,
    step_minutes: int,
    token_cfg: TokenConfig,
    decode_cfg: DecodeConfig,
    device: torch.device,
    seed: int,
) -> tuple[float, list[dict], dict]:
    context = list(prompt_ids)
    continuation: list[int] = []
    eos_id = vocab.id("<eos>")
    for index, action in enumerate(actions):
        minute = cut_minute + index * step_minutes
        if action != "ACTION::NOOP":
            action_id = vocab.id(action)
            context.append(action_id)
            continuation.append(action_id)
        generated = generate_continuation(
            model,
            vocab,
            context,
            minute,
            minute + step_minutes,
            token_cfg=token_cfg,
            decode_cfg=decode_cfg,
            device=device,
            seed=seed + minute,
        )
        generated = [token for token in generated if token != eos_id]
        context.extend(generated)
        continuation.extend(generated)
    end_minute = cut_minute + len(actions) * step_minutes
    frames = decode_frames(vocab, continuation, start_minute=cut_minute)
    metrics = branch_metrics(frames, cut_minute, end_minute)
    return metrics["home_possession"], frames, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate timing candidates with rollout sanity filtering.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rollout-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--early-stop-delta-pp", type=float, default=None)
    parser.add_argument("--early-stop-scope", choices=("accepted", "all"), default="accepted")
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=DecodeConfig.prompt_tail_tokens)
    parser.add_argument("--prefix-possession-target", action="store_true")
    parser.add_argument("--possession-lookback-minutes", type=int, default=15)
    parser.add_argument("--external-possession-weight", type=float, default=0.0)
    parser.add_argument("--possession-target-calibration-json", default=None)
    parser.add_argument("--density-from-prefix", action="store_true")
    parser.add_argument("--density-lookback-minutes", type=int, default=15)
    parser.add_argument("--density-multiplier", type=float, default=1.0)
    parser.add_argument("--event-density-tolerance", type=int, default=DecodeConfig.event_density_tolerance)
    parser.add_argument("--conditional-prior-json", default=None)
    parser.add_argument("--conditional-prior-bias-strength", type=float, default=0.0)
    parser.add_argument("--no-organic-actions", action="store_true")
    parser.add_argument("--possession-guidance-prior-events", type=float, default=0.0)
    parser.add_argument("--reject-possession-below", type=float, default=0.02)
    parser.add_argument("--reject-possession-above", type=float, default=0.98)
    parser.add_argument("--min-possession-frames", type=int, default=8)
    parser.add_argument("--min-events-per-minute", type=float, default=10.0)
    parser.add_argument("--max-events-per-minute", type=float, default=80.0)
    parser.add_argument("--min-accepted-seeds", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    conditional_prior = load_json(args.conditional_prior_json) if args.conditional_prior_json else None
    possession_target_calibration = (
        load_json(args.possession_target_calibration_json)
        if args.possession_target_calibration_json
        else None
    )

    source = load_json(args.candidates_json)
    candidates = select_candidates(source, args.top_k)
    results = []
    for candidate_index, candidate in enumerate(candidates):
        match_json = candidate["match_json"]
        cut_minute = int(candidate["cut_minute"])
        end_minute = int(candidate.get("end_minute", cut_minute + args.step_minutes))
        step_minutes = int(candidate.get("step_minutes", args.step_minutes))
        action = candidate.get("action") or source.get("anchor", {}).get("action")
        n_steps = max(1, (end_minute - cut_minute) // step_minutes)
        policy_actions = [action] + ["ACTION::NOOP"] * (n_steps - 1)
        noop_actions = ["ACTION::NOOP"] * n_steps

        events = load_json(match_json)
        ids, frames = encoder.encode_match(events, None)
        prompt_ids = ids[: token_index_at_cut(frames, cut_minute)]
        target_home_possession = None
        if args.prefix_possession_target:
            target_home_possession = prefix_home_possession_target(
                frames, cut_minute, args.possession_lookback_minutes
            )
            if target_home_possession is not None and possession_target_calibration:
                full_prefix_frames = [
                    frame for frame in frames
                    if 0 <= frame.get("minute", 0) < cut_minute
                ]
                full_prefix_possession = home_possession_ratio(full_prefix_frames)
                target_home_possession = (
                    float(possession_target_calibration.get("intercept", 0.0))
                    + float(possession_target_calibration.get("prefix_coef", 1.0)) * float(target_home_possession)
                    + float(possession_target_calibration.get("full_prefix_coef", 0.0)) * float(full_prefix_possession)
                )
                target_home_possession = max(0.05, min(0.95, target_home_possession))
        target_events_per_minute = None
        if args.density_from_prefix:
            lookback_start = max(0, cut_minute - args.density_lookback_minutes)
            density_prefix_frames = [
                frame for frame in frames
                if lookback_start <= frame.get("minute", 0) < cut_minute
            ]
            target_events_per_minute = (
                len(density_prefix_frames) / max(1, cut_minute - lookback_start)
            ) * max(0.1, float(args.density_multiplier))
        decode_cfg = make_decode_cfg(
            args,
            target_home_possession=target_home_possession,
            target_events_per_minute=target_events_per_minute,
            conditional_prior=conditional_prior,
        )
        rows = []
        accepted_rows = []
        rejected_rows = []
        stopped_early = False
        for seed_index in range(args.rollout_seeds):
            seed = args.seed + candidate_index * 10000 + seed_index * 1000
            noop, _, noop_metrics = replay_with_metrics(
                model, vocab, prompt_ids, noop_actions, cut_minute, step_minutes,
                token_cfg, decode_cfg, device, seed,
            )
            policy, _, policy_metrics = replay_with_metrics(
                model, vocab, prompt_ids, policy_actions, cut_minute, step_minutes,
                token_cfg, decode_cfg, device, seed,
            )
            noop_reasons = sanity_reasons(noop_metrics, args)
            policy_reasons = sanity_reasons(policy_metrics, args)
            row = {
                "seed": seed,
                "noop_possession": noop,
                "policy_possession": policy,
                "delta_pp": (policy - noop) * 100.0,
                "noop_metrics": noop_metrics,
                "policy_metrics": policy_metrics,
                "accepted": not noop_reasons and not policy_reasons,
                "noop_reject_reasons": noop_reasons,
                "policy_reject_reasons": policy_reasons,
            }
            rows.append(row)
            if row["accepted"]:
                accepted_rows.append(row)
            else:
                rejected_rows.append(row)
            print(
                f"[progress] candidate={candidate_index + 1}/{len(candidates)} "
                f"match={Path(match_json).stem} minute={cut_minute} "
                f"seed={seed_index + 1}/{args.rollout_seeds} delta_pp={row['delta_pp']:.3f} "
                f"accepted={row['accepted']}",
                flush=True,
            )
            if (
                args.early_stop_delta_pp is not None
                and (row["accepted"] or args.early_stop_scope == "all")
                and row["delta_pp"] <= args.early_stop_delta_pp
            ):
                stopped_early = True
                print(
                    f"[early_stop] candidate={candidate_index + 1}/{len(candidates)} "
                    f"match={Path(match_json).stem} minute={cut_minute} "
                    f"delta_pp={row['delta_pp']:.3f} threshold={args.early_stop_delta_pp:.3f} scope={args.early_stop_scope}",
                    flush=True,
                )
                break

        accepted_deltas = [row["delta_pp"] for row in accepted_rows]
        all_deltas = [row["delta_pp"] for row in rows]
        result = {
            **candidate,
            "match": Path(match_json).stem,
            "actions": policy_actions,
            "rollout_seeds": args.rollout_seeds,
            "completed_seeds": len(rows),
            "accepted_seeds": len(accepted_rows),
            "rejected_seeds": len(rejected_rows),
            "stopped_early": stopped_early,
            "early_stop_delta_pp": args.early_stop_delta_pp,
            "target_home_possession": target_home_possession,
            "all_seed_summary": summarize(all_deltas),
            "accepted_seed_summary": summarize(accepted_deltas),
            "passes_min_accepted_seeds": len(accepted_rows) >= args.min_accepted_seeds,
            "rows": rows,
        }
        results.append(result)
        save_json(
            {
                "complete": False,
                "checkpoint": args.checkpoint,
                "candidates_json": args.candidates_json,
                "completed_candidates": len(results),
                "sanity_filter": vars(args),
                "results": results,
            },
            args.out_json,
        )

    accepted_deltas = [row["delta_pp"] for result in results for row in result["rows"] if row["accepted"]]
    all_deltas = [row["delta_pp"] for result in results for row in result["rows"]]
    save_json(
        {
            "complete": True,
            "checkpoint": args.checkpoint,
            "candidates_json": args.candidates_json,
            "completed_candidates": len(results),
            "sanity_filter": vars(args),
            "overall_all_seeds": summarize(all_deltas),
            "overall_accepted_seeds": summarize(accepted_deltas),
            "results": results,
        },
        args.out_json,
    )


if __name__ == "__main__":
    main()
