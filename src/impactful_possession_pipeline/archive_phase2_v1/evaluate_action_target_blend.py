import argparse
import dataclasses
import math
from pathlib import Path

import torch

from .config import DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, load_json, mean, save_json, std
from .vocabulary import Vocab


def sem_lcb(values: list[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def cvar_bottom(values: list[float], fraction: float = 0.25) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * fraction))
    return mean(ordered[:count])


@torch.no_grad()
def predicted_rest_possession(model, ids: list[int], device: torch.device) -> float:
    context = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(context, dtype=torch.bool)
    return float(model.predict_values(context, mask)["possession"][0, 2].detach().cpu())


def decode_cfg_with_target(base: DecodeConfig, target: float | None, external_weight: float) -> DecodeConfig:
    return dataclasses.replace(
        base,
        target_home_possession=target,
        external_possession_weight=external_weight if target is not None else 0.0,
    )


def replay_target_blend(
    model,
    vocab: Vocab,
    prompt_ids: list[int],
    actions: list[str],
    cut_minute: int,
    step_minutes: int,
    token_cfg: TokenConfig,
    base_decode_cfg: DecodeConfig,
    device: torch.device,
    seed: int,
    target_blend: float,
    max_target_delta_pp: float,
    external_weight: float,
) -> tuple[float, list[dict]]:
    context = list(prompt_ids)
    continuation: list[int] = []
    eos_id = vocab.id("<eos>")
    target_rows = []
    for index, action in enumerate(actions):
        minute = cut_minute + index * step_minutes
        base_target = predicted_rest_possession(model, context, device)
        target = None
        if action != "ACTION::NOOP":
            action_id = vocab.id(action)
            context.append(action_id)
            continuation.append(action_id)
            action_target = predicted_rest_possession(model, context, device)
            delta = action_target - base_target
            max_delta = max_target_delta_pp / 100.0
            delta = max(-max_delta, min(max_delta, delta))
            target = max(0.02, min(0.98, base_target + target_blend * delta))
            target_rows.append(
                {
                    "step_index": index,
                    "minute": minute,
                    "action": action,
                    "base_target": base_target,
                    "action_target": action_target,
                    "blended_target": target,
                    "raw_target_delta_pp": (action_target - base_target) * 100.0,
                    "used_target_delta_pp": (target - base_target) * 100.0,
                }
            )
        generated = generate_continuation(
            model,
            vocab,
            context,
            minute,
            minute + step_minutes,
            token_cfg=token_cfg,
            decode_cfg=decode_cfg_with_target(base_decode_cfg, target, external_weight),
            device=device,
            seed=seed + minute,
        )
        generated = [token for token in generated if token != eos_id]
        context.extend(generated)
        continuation.extend(generated)
    frames = decode_frames(vocab, continuation, start_minute=cut_minute)
    return home_possession_ratio(frames), target_rows


def summarize(deltas: list[float]) -> dict:
    return {
        "n": len(deltas),
        "mean_delta_pp": mean(deltas),
        "std_delta_pp": std(deltas),
        "lcb90_delta_pp": sem_lcb(deltas),
        "cvar25_delta_pp": cvar_bottom(deltas),
        "worst_delta_pp": min(deltas) if deltas else 0.0,
        "positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in deltas]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conservative action-target blending for intervention rollouts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--match-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="ACTION::HOME_SUB_ROLE_CENTRAL_MIDFIELDER")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=50)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--rollout-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=97000)
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--prompt-tail-tokens", type=int, default=DecodeConfig.prompt_tail_tokens)
    parser.add_argument("--target-events-per-minute", type=float, default=None)
    parser.add_argument("--event-density-tolerance", type=int, default=DecodeConfig.event_density_tolerance)
    parser.add_argument("--possession-guidance-strength", type=float, default=2.0)
    parser.add_argument("--possession-guidance-prior-events", type=float, default=80.0)
    parser.add_argument("--target-blend", type=float, default=0.50)
    parser.add_argument("--max-target-delta-pp", type=float, default=6.0)
    parser.add_argument("--external-possession-weight", type=float, default=1.0)
    parser.add_argument("--stop-if-delta-below", type=float, default=None)
    parser.add_argument("--min-seeds-before-stop", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    events = load_json(args.match_json)
    ids, frames = encoder.encode_match(events, None)
    prompt_ids = ids[: token_index_at_cut(frames, args.cut_minute)]
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

    rows = []
    for seed_index in range(args.rollout_seeds):
        seed = args.seed + seed_index * 1000
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
        save_json({"complete": False, "rows": rows}, args.out_json)
        print(f"[progress] {seed_index + 1}/{args.rollout_seeds} delta_pp={row['delta_pp']:.3f}", flush=True)
        if (
            args.stop_if_delta_below is not None
            and len(rows) >= args.min_seeds_before_stop
            and row["delta_pp"] <= args.stop_if_delta_below
        ):
            deltas = [item["delta_pp"] for item in rows]
            save_json(
                {
                    "complete": False,
                    "early_stopped": True,
                    "stop_reason": f"delta_pp <= {args.stop_if_delta_below}",
                    "checkpoint": args.checkpoint,
                    "match_json": args.match_json,
                    "action": args.action,
                    "actions": policy_actions,
                    "cut_minute": args.cut_minute,
                    "end_minute": args.end_minute,
                    "step_minutes": args.step_minutes,
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
                    **summarize(deltas),
                    "rows": rows,
                },
                args.out_json,
            )
            print(f"[early_stop] delta_pp={row['delta_pp']:.3f} threshold={args.stop_if_delta_below:.3f}", flush=True)
            return

    deltas = [row["delta_pp"] for row in rows]
    save_json(
        {
            "complete": True,
            "checkpoint": args.checkpoint,
            "match_json": args.match_json,
            "action": args.action,
            "actions": policy_actions,
            "cut_minute": args.cut_minute,
            "end_minute": args.end_minute,
            "step_minutes": args.step_minutes,
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
            **summarize(deltas),
            "rows": rows,
        },
        args.out_json,
    )
    print(summarize(deltas), flush=True)


if __name__ == "__main__":
    main()
