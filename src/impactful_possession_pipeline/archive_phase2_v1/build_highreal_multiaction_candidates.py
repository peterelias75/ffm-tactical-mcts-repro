import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DataConfig
from .train_context_intervention_scorer import (
    action_name,
    checkpoint_event_map,
    context_features,
    dedupe_rows,
    expand_inputs,
    fit_ridge,
    load_label_rows,
    make_matrix,
    rows_with_features,
    standardize,
)
from .train_intervention_risk_gate import checkpoint_test_files, parse_minutes
from .utils import load_json, save_json


DEFAULT_ACTIONS = [
    "HOME_SUB_ROLE_WINGER",
    "HOME_SUB_ROLE_FORWARD",
    "HOME_SUB_ROLE_CENTRAL_MIDFIELDER",
    "HOME_TACTIC_FORM_433",
]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def sem_lcb(values: Sequence[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def min_distance(vector: np.ndarray, examples: Sequence[np.ndarray]) -> float:
    if not examples:
        return 0.0
    return min(float(np.linalg.norm(vector - example)) for example in examples)


def candidate_rows(checkpoint: str, event_by_name: dict[str, str], actions: list[str], minutes: list[int], max_matches: int) -> list[dict]:
    rows = []
    for match_json in checkpoint_test_files(checkpoint)[:max_matches]:
        event_file = event_by_name.get(Path(match_json).name, match_json)
        if not Path(event_file).exists():
            continue
        events = load_json(event_file)
        for minute in minutes:
            for action in actions:
                base_row = {
                    "value_head_delta_pp": 0.0,
                    "mean_noop_home_possession": 0.5,
                    "action": action,
                }
                rows.append(
                    {
                        "match": Path(event_file).name,
                        "match_json": event_file,
                        "action": f"ACTION::{action}",
                        "cut_minute": int(minute),
                        "end_minute": int(minute) + 5,
                        "features": context_features(events, action, base_row, int(minute)),
                    }
                )
    return rows


def strip_features(rows: list[dict]) -> list[dict]:
    clean = []
    for row in rows:
        item = dict(row)
        item.pop("features", None)
        clean.append(item)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-realism multi-action validation candidates from paired rollout labels.")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--actions", nargs="*", default=DEFAULT_ACTIONS)
    parser.add_argument("--candidate-minutes", default="45,50,55,60")
    parser.add_argument("--max-candidate-matches", type=int, default=500)
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--min-predicted-delta-pp", type=float, default=-0.5)
    parser.add_argument("--min-risk-score", type=float, default=0.0)
    parser.add_argument("--min-robust-lcb-pp", type=float, default=0.0)
    parser.add_argument("--min-robust-positive-rate", type=float, default=0.75)
    parser.add_argument("--min-robust-seeds", type=int, default=4)
    parser.add_argument("--boundary-max-lcb-pp", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=24)
    args = parser.parse_args()

    label_paths = expand_inputs(args.labels)
    raw_rows = load_label_rows(label_paths)
    deduped_rows = dedupe_rows(raw_rows)
    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    label_rows, names = rows_with_features(deduped_rows, event_by_name, 45)
    if len(label_rows) < 10:
        raise RuntimeError(f"not enough high-realism labels: {len(label_rows)}")

    train_x_raw = make_matrix(label_rows, names)
    train_x, mean_x, std_x = standardize(train_x_raw, train_x_raw, names)
    train_y = np.asarray([row["mean_rollout_delta_pp"] for row in label_rows], dtype=np.float64)
    weights = fit_ridge(train_x, train_y, args.ridge_alpha)
    for index, row in enumerate(label_rows):
        row["_vector"] = train_x[index]
        row["lcb90_pp"] = sem_lcb([float(value) for value in row["seed_deltas_pp"]])

    robust_rows = [
        row
        for row in label_rows
        if row["n_seeds"] >= args.min_robust_seeds
        and row["lcb90_pp"] >= args.min_robust_lcb_pp
        and row["positive_seed_rate"] >= args.min_robust_positive_rate
    ]
    boundary_rows = [row for row in label_rows if row["lcb90_pp"] < args.boundary_max_lcb_pp]
    robust_vectors = [row["_vector"] for row in robust_rows]
    boundary_vectors = [row["_vector"] for row in boundary_rows]

    seen = {(row["match"], f"ACTION::{action_name(row['action'])}", row.get("cut_minute")) for row in label_rows}
    candidates = candidate_rows(
        args.checkpoint,
        event_by_name,
        args.actions,
        parse_minutes(args.candidate_minutes),
        args.max_candidate_matches,
    )
    candidates = [row for row in candidates if (row["match"], row["action"], row["cut_minute"]) not in seen]
    cand_x_raw = make_matrix(candidates, names)
    cand_x = (cand_x_raw - mean_x) / std_x
    for row, pred, vector in zip(candidates, cand_x @ weights, cand_x):
        robust_distance = min_distance(vector, robust_vectors)
        boundary_distance = min_distance(vector, boundary_vectors)
        margin = boundary_distance - robust_distance
        row["predicted_delta_pp"] = float(pred)
        row["robust_distance"] = robust_distance
        row["boundary_distance"] = boundary_distance
        row["distance_margin"] = margin
        row["risk_score"] = float(pred) + 2.0 * margin - 0.75 * robust_distance

    candidates = sorted(candidates, key=lambda row: (row["risk_score"], row["predicted_delta_pp"]), reverse=True)
    recommended = [
        row
        for row in candidates
        if row["predicted_delta_pp"] >= args.min_predicted_delta_pp
        and row["risk_score"] >= args.min_risk_score
        and row["distance_margin"] > 0.0
    ][: args.top_k]
    output = {
        "selector": "highreal_multiaction_risk_ridge",
        "label_inputs": [str(path) for path in label_paths],
        "label_rows": len(label_rows),
        "robust_rows": len(robust_rows),
        "boundary_rows": len(boundary_rows),
        "actions": args.actions,
        "candidate_minutes": parse_minutes(args.candidate_minutes),
        "max_candidate_matches": args.max_candidate_matches,
        "ridge_alpha": args.ridge_alpha,
        "min_predicted_delta_pp": args.min_predicted_delta_pp,
        "min_risk_score": args.min_risk_score,
        "recommended_validation": strip_features(recommended),
        "top_candidates": strip_features(candidates[: args.top_k]),
        "decision": "validate_recommended_candidates" if recommended else "no_predicted_positive_candidates",
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "label_rows": output["label_rows"],
        "robust_rows": output["robust_rows"],
        "boundary_rows": output["boundary_rows"],
        "recommended_validation": len(output["recommended_validation"]),
        "decision": output["decision"],
    }, indent=2))
    for row in output["top_candidates"][:10]:
        print(
            f"{row['match']} minute={row['cut_minute']} action={row['action']} "
            f"risk={row['risk_score']:.3f} pred={row['predicted_delta_pp']:.3f} "
            f"margin={row['distance_margin']:.3f}"
        )


if __name__ == "__main__":
    main()
