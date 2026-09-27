import argparse

import torch

from .config import DecodeConfig, PlannerConfig, TokenConfig
from .dataset import token_index_at_cut
from .model import load_checkpoint
from .planner import plan_second_half
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, save_json
from .vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan possession-improving home-team interventions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--match-json", required=True)
    parser.add_argument("--lineup-json", default=None)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/intervention_plan.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-seeds", type=int, default=PlannerConfig.rollout_seeds)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=90)
    parser.add_argument("--step-minutes", type=int, default=PlannerConfig.step_minutes)
    parser.add_argument("--allowed-actions", default="", help="Comma-separated action-token allowlist.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    events = load_json(args.match_json)
    ids, frames = encoder.encode_match(events, args.lineup_json)
    prompt_ids = ids[:token_index_at_cut(frames, args.cut_minute)]
    planner_cfg = PlannerConfig(rollout_seeds=args.rollout_seeds, step_minutes=args.step_minutes)
    allowed_actions = [item.strip() for item in args.allowed_actions.split(",") if item.strip()] or None
    result = plan_second_half(
        model,
        vocab,
        prompt_ids,
        token_cfg=token_cfg,
        decode_cfg=DecodeConfig(),
        planner_cfg=planner_cfg,
        device=device,
        seed=args.seed,
        cut_minute=args.cut_minute,
        end_minute=args.end_minute,
        allowed_actions=allowed_actions,
    )
    result["match_json"] = args.match_json
    save_json(result, args.out_json)
    print(f"saved intervention plan to {args.out_json}")
    print(f"simulated_home_possession={result['simulated_home_possession']:.3f}")


if __name__ == "__main__":
    main()
