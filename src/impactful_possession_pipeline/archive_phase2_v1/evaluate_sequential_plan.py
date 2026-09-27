import argparse
from pathlib import Path

import torch

from .config import DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, load_json, mean, save_json, std
from .vocabulary import Vocab


def replay(
    model,
    vocab,
    prompt_ids,
    actions,
    cut_minute,
    step_minutes,
    token_cfg,
    decode_cfg,
    device,
    seed,
):
    context = list(prompt_ids)
    continuation = []
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
    frames = decode_frames(vocab, continuation, start_minute=cut_minute)
    return home_possession_ratio(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired validation of a sequential intervention plan.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--match-json", required=True)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--lineup-json", default=None)
    parser.add_argument("--rollout-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--step-minutes", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    ids, frames = encoder.encode_match(load_json(args.match_json), args.lineup_json)
    prompt_ids = ids[:token_index_at_cut(frames, args.cut_minute)]
    plan = load_json(args.plan_json)
    actions = [item["chosen_action"] for item in plan["interventions"]]
    noop_actions = ["ACTION::NOOP"] * len(actions)
    rows = []
    for index in range(args.rollout_seeds):
        seed = args.seed + index * 1000
        noop = replay(
            model, vocab, prompt_ids, noop_actions, args.cut_minute, args.step_minutes,
            token_cfg, DecodeConfig(), device, seed,
        )
        policy = replay(
            model, vocab, prompt_ids, actions, args.cut_minute, args.step_minutes,
            token_cfg, DecodeConfig(), device, seed,
        )
        rows.append({
            "seed": seed,
            "noop_possession": noop,
            "policy_possession": policy,
            "delta_pp": (policy - noop) * 100.0,
        })
        save_json({"complete": False, "rows": rows}, args.out_json)
        print(f"[progress] {index + 1}/{args.rollout_seeds} delta_pp={rows[-1]['delta_pp']:.3f}", flush=True)

    deltas = [row["delta_pp"] for row in rows]
    result = {
        "complete": True,
        "checkpoint": args.checkpoint,
        "match_json": args.match_json,
        "plan_json": args.plan_json,
        "actions": actions,
        "rollout_seeds": args.rollout_seeds,
        "mean_delta_pp": mean(deltas),
        "std_delta_pp": std(deltas),
        "positive_seed_rate": mean([1.0 if value > 0 else 0.0 for value in deltas]),
        "rows": rows,
    }
    save_json(result, args.out_json)
    print(result, flush=True)


if __name__ == "__main__":
    main()
