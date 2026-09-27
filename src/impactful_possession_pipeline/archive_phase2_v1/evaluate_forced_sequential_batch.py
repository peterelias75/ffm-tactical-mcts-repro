import argparse
import math
from pathlib import Path

import torch

from .config import DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .evaluate_sequential_plan import replay
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, mean, save_json, std
from .vocabulary import Vocab


def sem_lcb(values: list[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def checkpoint_test_files(checkpoint: str) -> list[str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    return list(ckpt.get("extra", {}).get("test_files", []))


def summarize(deltas: list[float]) -> dict:
    return {
        "n": len(deltas),
        "mean_delta_pp": mean(deltas),
        "std_delta_pp": std(deltas),
        'lcb90_delta_pp': sem_lcb(deltas),
        "positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in deltas]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fixed first-step sequential intervention over a holdout batch.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--match-jsons", nargs="*", default=None)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--max-matches", type=int, default=8)
    parser.add_argument("--action", default="ACTION::HOME_SUB_NEUTRAL")
    parser.add_argument("--rollout-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=60)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=DecodeConfig.prompt_tail_tokens)
    parser.add_argument("--possession-guidance-strength", type=float, default=DecodeConfig.possession_guidance_strength)
    parser.add_argument("--no-organic-actions", action="store_true")
    parser.add_argument("--enforce-possession-state", action="store_true")
    parser.add_argument("--football-structure-constraints", action="store_true")
    parser.add_argument("--pass-receipt-boost", type=float, default=DecodeConfig.pass_receipt_boost)
    parser.add_argument("--pass-carry-boost", type=float, default=DecodeConfig.pass_carry_boost)
    parser.add_argument("--overproduced-event-penalty", type=float, default=DecodeConfig.overproduced_event_penalty)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    decode_cfg = DecodeConfig(
        top_p=args.top_p,
        temperature=args.temperature,
        prompt_tail_tokens=args.prompt_tail_tokens,
        possession_guidance_strength=args.possession_guidance_strength,
        include_organic_actions=not args.no_organic_actions,
        enforce_possession_state=args.enforce_possession_state,
        football_structure_constraints=args.football_structure_constraints,
        pass_receipt_boost=args.pass_receipt_boost,
        pass_carry_boost=args.pass_carry_boost,
        overproduced_event_penalty=args.overproduced_event_penalty,
    )

    match_jsons = args.match_jsons or checkpoint_test_files(args.checkpoint)
    match_jsons = match_jsons[args.test_offset : args.test_offset + args.max_matches]
    n_steps = max(1, (args.end_minute - args.cut_minute) // args.step_minutes)
    policy_actions = [args.action] + ["ACTION::NOOP"] * (n_steps - 1)
    noop_actions = ["ACTION::NOOP"] * n_steps

    results = []
    for match_index, match_json in enumerate(match_jsons):
        events = load_json(match_json)
        ids, frames = encoder.encode_match(events, None)
        prompt_ids = ids[: token_index_at_cut(frames, args.cut_minute)]
        rows = []
        for seed_index in range(args.rollout_seeds):
            seed = args.seed + match_index * 10000 + seed_index * 1000
            noop = replay(
                model,
                vocab,
                prompt_ids,
                noop_actions,
                args.cut_minute,
                args.step_minutes,
                token_cfg,
                decode_cfg,
                device,
                seed,
            )
            policy = replay(
                model,
                vocab,
                prompt_ids,
                policy_actions,
                args.cut_minute,
                args.step_minutes,
                token_cfg,
                decode_cfg,
                device,
                seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "noop_possession": noop,
                    "policy_possession": policy,
                    "delta_pp": (policy - noop) * 100.0,
                }
            )
            print(
                f"[progress] match={Path(match_json).stem} {seed_index + 1}/{args.rollout_seeds} "
                f"delta_pp={rows[-1]['delta_pp']:.3f}",
                flush=True,
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
        all_deltas = [row["delta_pp"] for result in results for row in result["rows"]]
        save_json(
            {
                "complete": False,
                "checkpoint": args.checkpoint,
                "action": args.action,
                "cut_minute": args.cut_minute,
                "end_minute": args.end_minute,
                "step_minutes": args.step_minutes,
                "decode_config": {
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                    "prompt_tail_tokens": args.prompt_tail_tokens,
                    "possession_guidance_strength": args.possession_guidance_strength,
                    "include_organic_actions": not args.no_organic_actions,
                    "enforce_possession_state": args.enforce_possession_state,
                    "football_structure_constraints": args.football_structure_constraints,
                    "pass_receipt_boost": args.pass_receipt_boost,
                    "pass_carry_boost": args.pass_carry_boost,
                    "overproduced_event_penalty": args.overproduced_event_penalty,
                },
                "test_offset": args.test_offset,
                "max_matches": args.max_matches,
                "completed_matches": len(results),
                "overall": summarize(all_deltas),
                "match_positive_rate": mean([1.0 if result["mean_delta_pp"] > 0.0 else 0.0 for result in results]),
                "results": results,
            },
            args.out_json,
        )

    all_deltas = [row["delta_pp"] for result in results for row in result["rows"]]
    save_json(
        {
            "complete": True,
            "checkpoint": args.checkpoint,
            "action": args.action,
            "cut_minute": args.cut_minute,
            "end_minute": args.end_minute,
            "step_minutes": args.step_minutes,
            "decode_config": {
                "top_p": args.top_p,
                "temperature": args.temperature,
                "prompt_tail_tokens": args.prompt_tail_tokens,
                "possession_guidance_strength": args.possession_guidance_strength,
                "include_organic_actions": not args.no_organic_actions,
                "enforce_possession_state": args.enforce_possession_state,
                "football_structure_constraints": args.football_structure_constraints,
                "pass_receipt_boost": args.pass_receipt_boost,
                "pass_carry_boost": args.pass_carry_boost,
                "overproduced_event_penalty": args.overproduced_event_penalty,
            },
            "test_offset": args.test_offset,
            "max_matches": args.max_matches,
            "completed_matches": len(results),
            "overall": summarize(all_deltas),
            "match_positive_rate": mean([1.0 if result["mean_delta_pp"] > 0.0 else 0.0 for result in results]),
            "results": results,
        },
        args.out_json,
    )


if __name__ == "__main__":
    main()
