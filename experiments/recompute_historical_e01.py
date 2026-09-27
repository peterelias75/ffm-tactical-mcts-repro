"""Independently recompute the historical FFM Table 1 row from raw CSVs."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from impactful_possession_pipeline.utils import mean, save_json


BALL_EVENTS = {"pass", "dribble", "ball receipt", "ball recovery", "shot", "carry"}


def parse_sections(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    sections = {"home": [], "away": []}
    active = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped in ("Home", "Away"):
            active = stripped.lower()
        elif stripped and active:
            sections[active].append(line)
    parsed = []
    for name in ("home", "away"):
        rows = {}
        for row in csv.DictReader(sections[name]):
            rows[row["Key"]] = {
                "GT": float(row["GT"]), "GEN": float(row["GEN"]),
                "Difference": float(row["Difference"]),
            }
        parsed.append(rows)
    return parsed[0], parsed[1]


def parse_zone(path: Path) -> list[list[float]]:
    lines = path.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == "DIFF") + 1
    matrix = []
    for line in lines[start:]:
        if not line.strip():
            break
        matrix.append([float(value) for value in line.split(",")])
    return matrix


def main() -> None:
    results_dir = Path("model_v3/results")
    all_dirs = sorted(path for path in results_dir.glob("*_results") if path.is_dir())
    selected = all_dirs[70:520]
    if len(all_dirs) != 600 or len(selected) != 450:
        raise RuntimeError(f"unexpected populations: all={len(all_dirs)} selected={len(selected)}")

    event_home, event_away, position_home, position_away = [], [], [], []
    zones, possession_errors = [], []
    integrity = hashlib.sha256()
    for directory in selected:
        paths = [directory / "events.csv", directory / "positions.csv", directory / "zones.csv"]
        if not all(path.is_file() for path in paths):
            raise RuntimeError(f"incomplete result directory: {directory}")
        for path in paths:
            integrity.update(str(path.relative_to(results_dir)).encode())
            integrity.update(path.read_bytes())
        home_events, away_events = parse_sections(paths[0])
        home_positions, away_positions = parse_sections(paths[1])
        event_home.append(home_events); event_away.append(away_events)
        position_home.append(home_positions); position_away.append(away_positions)
        zones.append(parse_zone(paths[2]))
        gt_home = sum(row["GT"] for key, row in home_events.items() if key.lower() in BALL_EVENTS)
        gt_away = sum(row["GT"] for key, row in away_events.items() if key.lower() in BALL_EVENTS)
        gen_home = sum(row["GEN"] for key, row in home_events.items() if key.lower() in BALL_EVENTS)
        gen_away = sum(row["GEN"] for key, row in away_events.items() if key.lower() in BALL_EVENTS)
        gt_share = 100.0 * gt_home / (gt_home + gt_away) if gt_home + gt_away else 0.0
        gen_share = 100.0 * gen_home / (gen_home + gen_away) if gen_home + gen_away else 0.0
        possession_errors.append(abs(gt_share - gen_share))

    def aggregate(dicts: list[dict]) -> dict[str, float]:
        keys = sorted({key for item in dicts for key in item})
        return {key: mean([item.get(key, {}).get("Difference", 0.0) for item in dicts]) for key in keys}

    eh, ea = aggregate(event_home), aggregate(event_away)
    ph, pa = aggregate(position_home), aggregate(position_away)
    rows, cols = len(zones[0]), len(zones[0][0])
    zone_mad = [[mean([matrix[row][col] for matrix in zones]) for col in range(cols)] for row in range(rows)]
    result = {
        "experiment_id": "E01_historical_FFM",
        "provenance": {
            "results_dir": str(results_dir), "num_available_dirs": len(all_dirs),
            "selection": "sorted directories [70:520]", "num_selected": len(selected),
            "selected_raw_csv_sha256": integrity.hexdigest(),
            "historical_aggregator": "model_v3/mad_summary.py",
            "historical_summary": "model_v3/mad_summary.txt",
        },
        "summary": {
            "num_matches": len(selected),
            "possession_mae_pp": mean(possession_errors),
            "passes_mae": mean([eh.get("pass", 0.0), ea.get("pass", 0.0)]),
            "shots_mae": mean([eh.get("shot", 0.0), ea.get("shot", 0.0)]),
            "dribbles_mae": mean([eh.get("dribble", 0.0), ea.get("dribble", 0.0)]),
            "total_events_mae_category_average": mean(list(eh.values()) + list(ea.values())),
            "total_position_mae_category_average": mean(list(ph.values()) + list(pa.values())),
            "total_zone_mae_cell_average": mean([value for row in zone_mad for value in row]),
        },
        "paper_claim": {"num_matches": 450, "possession_mae_pp": 11.95, "passes_mae": 71.27, "shots_mae": 4.34, "dribbles_mae": 5.52, "total_events_mae": 11.04},
        "metric_caveat": "Historical Total Event MAD is the mean per-category event MAD across home and away, not absolute error in the aggregate event count.",
        "event_home_mad": eh, "event_away_mad": ea, "zone_mad": zone_mad,
        "selected_matches": [path.name for path in selected],
    }
    save_json(result, "paper_reproduction/runs/e01_historical_ffm/results.json")
    print(result["summary"])


if __name__ == "__main__":
    main()
