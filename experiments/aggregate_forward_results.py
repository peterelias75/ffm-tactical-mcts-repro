"""Combine disjoint forward-evaluation shards and verify match uniqueness."""

import argparse
from pathlib import Path

from impactful_possession_pipeline.utils import load_json, mean, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows = []
    shards = []
    for name in args.input:
        payload = load_json(name)
        shards.append({"path": name, "summary": payload.get("summary", {})})
        rows.extend(payload.get("matches", []))
    match_names = [row["match"] for row in rows]
    duplicates = sorted({name for name in match_names if match_names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"overlapping shards: {duplicates[:10]}")
    rows.sort(key=lambda row: row["match"])
    summary = {
        "num_matches": len(rows),
        "no_intervention_mae_pp": mean([row["abs_error_pp"] for row in rows]),
        "mean_generated_frame_count": mean([row["generated_frame_count"] for row in rows]),
        "passes_mae": mean([row["passes_abs_error"] for row in rows]),
        "shots_mae": mean([row["shots_abs_error"] for row in rows]),
        "dribbles_mae": mean([row["dribbles_abs_error"] for row in rows]),
        "total_events_mae": mean([row["total_events_abs_error"] for row in rows]),
        "all_shards_complete": all(item["summary"].get("complete", False) for item in shards),
        "paper_ffm_target_mae_pp": 11.95,
        "paper_target_num_matches": 450,
        "comparison_note": "Checkpoint has 340 persisted leakage-safe held-out matches; this run does not claim N=450 parity.",
    }
    save_json({"summary": summary, "shards": shards, "matches": rows}, Path(args.out_json))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
