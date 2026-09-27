import argparse
from pathlib import Path

import torch

from .config import DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, load_json, save_json
from .vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a structured continuation from a StatsBomb match prefix.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--match-json", required=True)
    parser.add_argument("--lineup-json", default=None)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=90)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/simulation.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, _ = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig()
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    events = load_json(args.match_json)
    ids, frames = encoder.encode_match(events, args.lineup_json)
    cut_idx = token_index_at_cut(frames, args.cut_minute)
    prompt_ids = ids[:cut_idx]
    generated = generate_continuation(
        model,
        vocab,
        prompt_ids,
        start_minute=args.cut_minute,
        end_minute=args.end_minute,
        token_cfg=token_cfg,
        decode_cfg=DecodeConfig(),
        device=device,
        seed=args.seed,
    )
    generated_frames = decode_frames(vocab, generated, start_minute=args.cut_minute)
    result = {
        "match_json": args.match_json,
        "cut_minute": args.cut_minute,
        "end_minute": args.end_minute,
        "generated_tokens": [vocab.token(i) for i in generated],
        "generated_frames": generated_frames,
        "simulated_home_possession": home_possession_ratio(generated_frames),
    }
    save_json(result, args.out_json)
    print(f"saved simulation to {args.out_json}")


if __name__ == "__main__":
    main()
