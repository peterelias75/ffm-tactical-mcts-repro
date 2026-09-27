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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mixed match/minute/action timing candidates.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rollout-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--early-stop-delta-pp", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=DecodeConfig.prompt_tail_tokens)
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
        rows = []
        stopped_early = False
        for seed_index in range(args.rollout_seeds):
            seed = args.seed + candidate_index * 10000 + seed_index * 1000
            noop = replay(
                model,
                vocab,
                prompt_ids,
                noop_actions,
                cut_minute,
                step_minutes,
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
                cut_minute,
                step_minutes,
                token_cfg,
                decode_cfg,
                device,
                seed,
            )
            row = {
                "seed": seed,
                "noop_possession": noop,
                "policy_possession": policy,
                "delta_pp": (policy - noop) * 100.0,
            }
            rows.append(row)
            print(
                f"[progress] candidate={candidate_index + 1}/{len(candidates)} "
                f"match={Path(match_json).stem} minute={cut_minute} "
                f"seed={seed_index + 1}/{args.rollout_seeds} delta_pp={row['delta_pp']:.3f}",
                flush=True,
            )

            if args.early_stop_delta_pp is not None and row["delta_pp"] <= args.early_stop_delta_pp:
                stopped_early = True
                print(
                    f"[early_stop] candidate={candidate_index + 1}/{len(candidates)} "
                    f"match={Path(match_json).stem} minute={cut_minute} "
                    f"delta_pp={row['delta_pp']:.3f} threshold={args.early_stop_delta_pp:.3f}",
                    flush=True,
                )
                break

        deltas = [row["delta_pp"] for row in rows]
        results.append(
            {
                **candidate,
                "match": Path(match_json).stem,
                "actions": policy_actions,
                "rollout_seeds": args.rollout_seeds,
                "completed_seeds": len(rows),
                "stopped_early": stopped_early,
                "early_stop_delta_pp": args.early_stop_delta_pp,
                **summarize(deltas),
                "rows": rows,
            }
        )
        all_deltas = [row["delta_pp"] for result in results for row in result["rows"]]
        save_json(
            {
                "complete": False,
                "checkpoint": args.checkpoint,
                "candidates_json": args.candidates_json,
                "completed_candidates": len(results),
                "early_stop_delta_pp": args.early_stop_delta_pp,
                "decode_config": {
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                    "prompt_tail_tokens": args.prompt_tail_tokens,
                },
                "overall": summarize(all_deltas),
                "results": results,
            },
            args.out_json,
        )

    all_deltas = [row["delta_pp"] for result in results for row in result["rows"]]
    save_json(
        {
            "complete": True,
            "checkpoint": args.checkpoint,
            "candidates_json": args.candidates_json,
            "completed_candidates": len(results),
            "early_stop_delta_pp": args.early_stop_delta_pp,
            "decode_config": {
                "top_p": args.top_p,
                "temperature": args.temperature,
                "prompt_tail_tokens": args.prompt_tail_tokens,
            },
            "overall": summarize(all_deltas),
            "results": results,
        },
        args.out_json,
    )


if __name__ == "__main__":
    main()
