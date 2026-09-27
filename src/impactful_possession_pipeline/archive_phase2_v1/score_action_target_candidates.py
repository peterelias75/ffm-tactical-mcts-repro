import argparse
import json
from pathlib import Path

import torch

from .config import TokenConfig
from .dataset import token_index_at_cut
from .evaluate_action_target_blend import predicted_rest_possession
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, save_json
from .vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate action target deltas with the value head.")
    parser.add_argument("--selector-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    selector = load_json(args.selector_json)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(selector["checkpoint"], device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    action_id_cache = {}
    event_cache = {}
    scored = []
    for index, candidate in enumerate(selector["top_candidates"][: args.top_n], start=1):
        match_json = candidate["match_json"]
        if match_json not in event_cache:
            event_cache[match_json] = load_json(match_json)
        ids, frames = encoder.encode_match(event_cache[match_json], None)
        prompt_ids = ids[: token_index_at_cut(frames, int(candidate["cut_minute"]))]
        base_target = predicted_rest_possession(model, prompt_ids, device)
        action = candidate["action"]
        if action not in action_id_cache:
            action_id_cache[action] = vocab.id(action)
        action_target = predicted_rest_possession(model, prompt_ids + [action_id_cache[action]], device)
        row = dict(candidate)
        row["selector_rank"] = index
        row["base_target"] = base_target
        row["action_target"] = action_target
        row["raw_target_delta_pp"] = (action_target - base_target) * 100.0
        scored.append(row)
        print(f"[score] {index}/{min(args.top_n, len(selector['top_candidates']))} {row['match']} {row['cut_minute']} raw_delta_pp={row['raw_target_delta_pp']:.3f}", flush=True)
    output = {
        "selector_json": args.selector_json,
        "checkpoint": selector["checkpoint"],
        "top_n": args.top_n,
        "positive_raw_delta": [row for row in scored if row["raw_target_delta_pp"] > 0.0],
        "scored_candidates": scored,
    }
    save_json(output, args.out_json)
    print(json.dumps({"scored": len(scored), "positive_raw_delta": len(output["positive_raw_delta"]), "top_positive": output["positive_raw_delta"][:5]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
