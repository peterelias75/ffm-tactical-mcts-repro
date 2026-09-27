"""Evaluate a persisted-holdout shard and save progress after every match."""

import argparse
import time
from pathlib import Path

import torch

from impactful_possession_pipeline.config import DecodeConfig, TokenConfig
from impactful_possession_pipeline.evaluate_pipeline import evaluate_match, resolve_test_files
from impactful_possession_pipeline.model import load_checkpoint
from impactful_possession_pipeline.statsbomb_features import StructuredStatsBombEncoder
from impactful_possession_pipeline.utils import load_json, mean, save_json
from impactful_possession_pipeline.vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default="/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/events")
    parser.add_argument("--lineups-dir", default="/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/lineups")
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # RTX 2080 Ti has 11,264 MiB. Keep roughly 1.8 GiB outside this
        # process even if a future configuration needs substantially more.
        torch.cuda.set_per_process_memory_fraction(0.84, device=0)
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    encoder = StructuredStatsBombEncoder(vocab, TokenConfig())
    proxy = argparse.Namespace(max_matches=10**9, events_dir=args.events_dir, seed=args.seed)
    all_test = resolve_test_files(extra, proxy)
    selected = all_test[args.offset : args.offset + args.count]
    out_path = Path(args.out_json)
    rows = load_json(out_path).get("matches", []) if args.resume and out_path.exists() else []
    required_metrics = {"passes_abs_error", "shots_abs_error", "dribbles_abs_error", "total_events_abs_error"}
    rows = [row for row in rows if required_metrics.issubset(row)]
    done = {row["match"] for row in rows}

    def persist(complete: bool) -> None:
        save_json({"summary": {
            "offset": args.offset, "num_requested": len(selected),
            "num_completed": len(rows), "complete": complete,
            "no_intervention_mae_pp": mean([r["abs_error_pp"] for r in rows]),
            "mean_generated_frame_count": mean([r["generated_frame_count"] for r in rows]),
            "mean_elapsed_seconds_per_match": mean([r.get("elapsed_seconds", 0.0) for r in rows]),
            "passes_mae": mean([r["passes_abs_error"] for r in rows]),
            "shots_mae": mean([r["shots_abs_error"] for r in rows]),
            "dribbles_mae": mean([r["dribbles_abs_error"] for r in rows]),
            "total_events_mae": mean([r["total_events_abs_error"] for r in rows]),
        }, "matches": rows}, out_path)

    for index, path in enumerate(selected):
        name = Path(path).name
        if name in done:
            continue
        started = time.time()
        row = evaluate_match(model, vocab, encoder, path, args.lineups_dir, device,
                             args.seed + args.offset + index, False, DecodeConfig())
        row["elapsed_seconds"] = time.time() - started
        rows.append(row)
        persist(False)
        print(f"{len(rows)}/{len(selected)} {name} error={row['abs_error_pp']:.3f}pp elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    persist(True)


if __name__ == "__main__":
    main()
