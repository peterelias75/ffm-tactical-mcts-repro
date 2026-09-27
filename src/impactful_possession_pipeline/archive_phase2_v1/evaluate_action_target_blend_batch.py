import argparse
import json
from pathlib import Path

import torch

from .config import DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .evaluate_action_target_blend import replay_target_blend, summarize
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, mean, save_json
from .vocabulary import Vocab


def checkpoint_test_files(checkpoint: str) -> list[str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    return list(ckpt.get("extra", {}).get("test_files", []))


def parse_match_jsons(args: argparse.Namespace) -> list[str]:
    if args.match_jsons:
        return args.match_jsons
    return checkpoint_test_files(args.checkpoint)[args.test_offset : args.test_offset + args.max_matches]


def summarize_matches(results: list[dict]) -> dict:
    deltas = [row["delta_pp"] for result in results for row in result["rows"]]
    summary = summarize(deltas)
    summary["match_count"] = len(results)
    summary["match_positive_rate"] = mean([1.0 if result["mean_delta_pp"] > 0.0 else 0.0 for result in results])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate action-target blending over a small match batch.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--match-jsons", nargs="*", default=None)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--max-matches", type=int, default=5)
    parser.add_argument("--action", default="ACTION::HOME_SUB_ROLE_CENTRAL_MIDFIELDER")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=50)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--rollout-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=DecodeConfig.prompt_tail_tokens)
    parser.add_argument("--target-events-per-minute", type=float, default=22.0)
    parser.add_argument("--event-density-tolerance", type=int, default=2)
    parser.add_argument("--possession-guidance-strength", type=float, default=2.0)
    parser.add_argument("--possession-guidance-prior-events", type=float, default=80.0)
    parser.add_argument("--target-blend", type=float, default=0.5)
    parser.add_argument("--max-target-delta-pp", type=float, default=6.0)
    parser.add_argument("--external-possession-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    match_jsons = parse_match_jsons(args)
    n_steps = max(1, (args.end_minute - args.cut_minute) // args.step_minutes)
    policy_actions = [args.action] + ["ACTION::NOOP"] * (n_steps - 1)
    noop_actions = ["ACTION::NOOP"] * n_steps
    base_decode_cfg = DecodeConfig(
        top_p=args.top_p,
        temperature=args.temperature,
        prompt_tail_tokens=args.prompt_tail_tokens,
        target_events_per_minute=args.target_events_per_minute,
        event_density_tolerance=args.event_density_tolerance,
        possession_guidance_strength=args.possession_guidance_strength,
        possession_guidance_prior_events=args.possession_guidance_prior_events,
    )

    results = []
    for match_index, match_json in enumerate(match_jsons):
        events = load_json(match_json)
        ids, frames = encoder.encode_match(events, None)
        prompt_ids = ids[: token_index_at_cut(frames, args.cut_minute)]
        rows = []
        for seed_index in range(args.rollout_seeds):
            seed = args.seed + match_index * 10000 + seed_index * 1000
            noop, noop_targets = replay_target_blend(
                model,
                vocab,
                prompt_ids,
                noop_actions,
                args.cut_minute,
                args.step_minutes,
                token_cfg,
                base_decode_cfg,
                device,
                seed,
                args.target_blend,
                args.max_target_delta_pp,
                args.external_possession_weight,
            )
            policy, policy_targets = replay_target_blend(
                model,
                vocab,
                prompt_ids,
                policy_actions,
                args.cut_minute,
                args.step_minutes,
                token_cfg,
                base_decode_cfg,
                device,
                seed,
                args.target_blend,
                args.max_target_delta_pp,
                args.external_possession_weight,
            )
            row = {
                "seed": seed,
                "noop_possession": noop,
                "policy_possession": policy,
                "delta_pp": (policy - noop) * 100.0,
                "noop_targets": noop_targets,
                "policy_targets": policy_targets,
            }
            rows.append(row)
            print(
                f"[progress] match={match_index + 1}/{len(match_jsons)} "
                f"{Path(match_json).stem} seed={seed_index + 1}/{args.rollout_seeds} "
                f"delta_pp={row['delta_pp']:.3f}",
                flush=True,
            )
            save_json(
                {
                    "complete": False,
                    "completed_matches": len(results),
                    "active_match": Path(match_json).stem,
                    "active_rows": rows,
                    "summary": summarize_matches(results) if results else {},
                    "results": results,
                },
                args.out_json,
            )
        deltas = [row["delta_pp"] for row in rows]
        results.append(
            {
                "match_json": match_json,
                "match": Path(match_json).stem,
                "actions": policy_actions,
                "rollout_seeds": args.rollout_seeds,
                **summarize(deltas),
                "rows": rows,
            }
        )
        save_json(
            {
                "complete": False,
                "checkpoint": args.checkpoint,
                "action": args.action,
                "decode_config": {
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                    "target_events_per_minute": args.target_events_per_minute,
                    "event_density_tolerance": args.event_density_tolerance,
                    "possession_guidance_strength": args.possession_guidance_strength,
                    "possession_guidance_prior_events": args.possession_guidance_prior_events,
                    "target_blend": args.target_blend,
                    "max_target_delta_pp": args.max_target_delta_pp,
                    "external_possession_weight": args.external_possession_weight,
                },
                "completed_matches": len(results),
                "summary": summarize_matches(results),
                "results": results,
            },
            args.out_json,
        )

    output = {
        "complete": True,
        "checkpoint": args.checkpoint,
        "action": args.action,
        "cut_minute": args.cut_minute,
        "end_minute": args.end_minute,
        "step_minutes": args.step_minutes,
        "rollout_seeds": args.rollout_seeds,
        "decode_config": {
            "top_p": args.top_p,
            "temperature": args.temperature,
            "prompt_tail_tokens": args.prompt_tail_tokens,
            "target_events_per_minute": args.target_events_per_minute,
            "event_density_tolerance": args.event_density_tolerance,
            "possession_guidance_strength": args.possession_guidance_strength,
            "possession_guidance_prior_events": args.possession_guidance_prior_events,
            "target_blend": args.target_blend,
            "max_target_delta_pp": args.max_target_delta_pp,
            "external_possession_weight": args.external_possession_weight,
        },
        "summary": summarize_matches(results),
        "results": results,
    }
    save_json(output, args.out_json)
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
