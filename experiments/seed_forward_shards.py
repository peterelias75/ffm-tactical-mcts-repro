"""Seed resumable forward-evaluation shards from compatible persisted rows.

Rows are reused only when their match filename occurs inside the exact target
slice of the checkpoint-persisted test split. The forward runner derives its
random seed from the absolute split index, so rows generated with seed 42 and
their original absolute offsets are deterministic-compatible.
"""

import argparse
from pathlib import Path

import torch

from impactful_possession_pipeline.utils import load_json, mean, save_json


REQUIRED = {
    "passes_abs_error", "shots_abs_error", "dribbles_abs_error",
    "total_events_abs_error", "abs_error_pp", "generated_frame_count",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--target", action="append", nargs=3, metavar=("OFFSET", "COUNT", "JSON"), required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    extra = checkpoint.get("extra", {})
    test_files = extra.get("test_files")
    if not test_files:
        raise RuntimeError("checkpoint has no persisted test_files")

    reusable = {}
    for source in args.source:
        for row in load_json(source).get("matches", []):
            if REQUIRED.issubset(row):
                reusable[row["match"]] = row

    report = []
    for offset_text, count_text, target_text in args.target:
        offset, count = int(offset_text), int(count_text)
        allowed = [Path(path).name for path in test_files[offset:offset + count]]
        rows = [reusable[name] for name in allowed if name in reusable]
        target = Path(target_text)
        summary = {
            "offset": offset,
            "num_requested": len(allowed),
            "num_completed": len(rows),
            "complete": len(rows) == len(allowed),
            "no_intervention_mae_pp": mean([r["abs_error_pp"] for r in rows]),
            "mean_generated_frame_count": mean([r["generated_frame_count"] for r in rows]),
            "mean_elapsed_seconds_per_match": 0.0,
            "passes_mae": mean([r["passes_abs_error"] for r in rows]),
            "shots_mae": mean([r["shots_abs_error"] for r in rows]),
            "dribbles_mae": mean([r["dribbles_abs_error"] for r in rows]),
            "total_events_mae": mean([r["total_events_abs_error"] for r in rows]),
            "seeded_from_compatible_persisted_rows": True,
        }
        save_json({"summary": summary, "matches": rows}, target)
        report.append({"target": str(target), "seeded": len(rows), "matches": [r["match"] for r in rows]})

    print(report)


if __name__ == "__main__":
    main()
