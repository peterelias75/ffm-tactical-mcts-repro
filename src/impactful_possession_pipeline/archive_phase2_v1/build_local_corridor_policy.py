import argparse
import json
import math
from pathlib import Path

from .utils import mean, save_json, std


def sem_lcb(values: list[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def load_rows(label_paths: list[Path], match: str, action: str) -> list[dict]:
    grouped: dict[int, dict] = {}
    for path in label_paths:
        data = json.loads(path.read_text())
        for row in data.get("matches", []):
            if row.get("match") != match:
                continue
            if row.get("action") != action:
                continue
            cut_minute = int(row["cut_minute"])
            item = grouped.setdefault(
                cut_minute,
                {
                    "match": match,
                    "match_json": row["match_json"],
                    "action": action,
                    "cut_minute": cut_minute,
                    "end_minute": int(row.get("end_minute", cut_minute + 5)),
                    "seed_deltas_pp": [],
                    "sources": [],
                },
            )
            item["seed_deltas_pp"].extend(float(value) for value in row.get("seed_deltas_pp", []))
            item["sources"].append(str(path))
    return sorted(grouped.values(), key=lambda row: row["cut_minute"])


def summarize_rows(rows: list[dict], min_lcb_pp: float, min_positive_seed_rate: float, min_seeds: int) -> list[dict]:
    summaries = []
    for row in rows:
        deltas = row["seed_deltas_pp"]
        positive_seed_rate = mean([1.0 if value > 0.0 else 0.0 for value in deltas])
        lcb = sem_lcb(deltas)
        status = (
            "robust_intervention_window"
            if len(deltas) >= min_seeds and lcb >= min_lcb_pp and positive_seed_rate >= min_positive_seed_rate
            else "stop_or_noop_boundary"
        )
        summaries.append(
            {
                **row,
                "n_seeds": len(deltas),
                "mean_rollout_delta_pp": mean(deltas),
                "std_rollout_delta_pp": std(deltas),
                "lcb90_rollout_delta_pp": lcb,
                "positive_seed_rate": positive_seed_rate,
                "status": status,
            }
        )
    return summaries


def contiguous_windows(rows: list[dict]) -> list[dict]:
    windows = []
    current: list[dict] = []
    for row in rows:
        if row["status"] != "robust_intervention_window":
            if current:
                windows.append(window_summary(current))
                current = []
            continue
        if current and row["cut_minute"] != current[-1]["cut_minute"] + 5:
            windows.append(window_summary(current))
            current = []
        current.append(row)
    if current:
        windows.append(window_summary(current))
    return windows


def window_summary(rows: list[dict]) -> dict:
    return {
        "start_minute": rows[0]["cut_minute"],
        "end_minute": rows[-1]["end_minute"],
        "action": rows[0]["action"],
        "cuts": [row["cut_minute"] for row in rows],
        "worst_lcb90_rollout_delta_pp": min(row["lcb90_rollout_delta_pp"] for row in rows),
        "worst_positive_seed_rate": min(row["positive_seed_rate"] for row in rows),
        "mean_of_means_delta_pp": mean([row["mean_rollout_delta_pp"] for row in rows]),
        "total_seed_count": sum(row["n_seeds"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a match-local sequential intervention corridor policy.")
    parser.add_argument("--label-jsons", nargs="+", required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-lcb-pp", type=float, default=0.0)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.75)
    parser.add_argument("--min-seeds", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows([Path(path) for path in args.label_jsons], args.match, args.action)
    summaries = summarize_rows(rows, args.min_lcb_pp, args.min_positive_seed_rate, args.min_seeds)
    windows = contiguous_windows(summaries)
    output = {
        "policy": "match_local_sequential_corridor",
        "match": args.match,
        "action": args.action,
        "thresholds": {
            "min_lcb_pp": args.min_lcb_pp,
            "min_positive_seed_rate": args.min_positive_seed_rate,
            "min_seeds": args.min_seeds,
        },
        "recommended_windows": windows,
        "policy_table": summaries,
        "decision": "use_local_corridor_for_case_study_validation" if windows else "default_to_noop",
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "match": output["match"],
        "recommended_windows": output["recommended_windows"],
        "decision": output["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
