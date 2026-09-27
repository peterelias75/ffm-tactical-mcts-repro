"""Free-running rollout MAE on the FULL pinned 25-match holdout, configurable decode knobs.

evaluate_pipeline.py evaluates against the checkpoint's persisted 340-match test
split and only exposes event-density decode flags. This instead reuses its
evaluate_match() over the pinned diagnostic holdout (the set the project's own
5pp gate is defined against) with a configurable temperature/top_p, so
different checkpoint candidates and decode settings can be compared on real
sample size instead of the noisy n=5 in-training check.
"""

import argparse
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, TokenConfig
from .evaluate_pipeline import evaluate_match
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, mean, save_json
from .vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--holdout-json", default="impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json")
    parser.add_argument("--max-holdout-matches", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=DecodeConfig.temperature)
    parser.add_argument("--top-p", type=float, default=DecodeConfig.top_p)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    decode_cfg = DecodeConfig(temperature=args.temperature, top_p=args.top_p)

    holdout = load_json(args.holdout_json)
    match_ids = holdout["match_ids"]
    if args.max_holdout_matches:
        match_ids = match_ids[: args.max_holdout_matches]

    rows = []
    for i, match_id in enumerate(match_ids):
        path = str(Path(args.events_dir) / match_id)
        row = evaluate_match(model, vocab, encoder, path, args.lineups_dir, device, args.seed + i, run_planner=False, decode_cfg=decode_cfg)
        rows.append(row)
        print(f"{i + 1}/{len(match_ids)} {row['match']} abs_error_pp={row['abs_error_pp']:.2f} sim_pos={row['no_intervention_home_possession']:.3f} real_pos={row['real_home_possession']:.3f}", flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "holdout_json": args.holdout_json,
        "num_matches": len(rows),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "no_intervention_mae_pp": mean([r["abs_error_pp"] for r in rows]),
        "mean_generated_frame_count": mean([r["generated_frame_count"] for r in rows]),
    }
    save_json({"summary": summary, "matches": rows}, args.out_json)
    print(f"saved to {args.out_json}")
    print(summary)


if __name__ == "__main__":
    main()
