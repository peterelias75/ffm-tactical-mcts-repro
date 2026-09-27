import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .build_corridor_aware_selector import (
    apply_standardize,
    feature_names,
    make_candidates,
    make_matrix,
    min_distance,
    robust_label,
    standardize,
    strip_vectors,
)
from .config import DataConfig
from .train_context_intervention_scorer import (
    action_name,
    checkpoint_event_map,
    context_features,
    dedupe_rows,
    expand_inputs,
    load_label_rows,
)
from .train_intervention_risk_gate import parse_minutes
from .utils import load_json, mean, save_json, std


DEFAULT_LABELS = [
    "impactful_possession_pipeline/runs/rich_actions_3400matches_finetune_3epochs_20260712_maxlen2048/labels_forced_*.json",
]


def sem_lcb(values: Sequence[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def add_label_features(rows: Sequence[dict], event_by_name: dict[str, str], default_cut_minute: int) -> list[dict]:
    featured = []
    for row in rows:
        event_file = event_by_name.get(row["match"])
        if not event_file or not Path(event_file).exists():
            continue
        cut_minute = int(row.get("cut_minute") or default_cut_minute)
        item = dict(row)
        item["cut_minute"] = cut_minute
        item["lcb90_pp"] = sem_lcb([float(value) for value in item["seed_deltas_pp"]])
        item["features"] = context_features(load_json(event_file), item["action"], item, cut_minute)
        featured.append(item)
    return featured


def seed_rows(label_rows: Sequence[dict]) -> list[dict]:
    rows = []
    for label in label_rows:
        for seed_index, delta in enumerate(label["seed_deltas_pp"]):
            item = dict(label)
            item["seed_index"] = seed_index
            item["seed_delta_pp"] = float(delta)
            item["seed_negative"] = 1.0 if float(delta) < 0.0 else 0.0
            item["seed_downside_pp"] = max(0.0, -float(delta))
            rows.append(item)
    return rows


def annotate_vectors(rows: Sequence[dict], names: Sequence[str], mu: np.ndarray, sigma: np.ndarray) -> None:
    x = apply_standardize(make_matrix(rows, names), mu, sigma)
    for index, row in enumerate(rows):
        row["_vector"] = x[index]


def score_candidates(
    candidates: list[dict],
    names: Sequence[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    weights: dict[str, np.ndarray],
    robust_rows: Sequence[dict],
    boundary_rows: Sequence[dict],
) -> None:
    x = apply_standardize(make_matrix(candidates, names), mu, sigma)
    robust_vectors = [row["_vector"] for row in robust_rows]
    boundary_vectors = [row["_vector"] for row in boundary_rows]
    for index, row in enumerate(candidates):
        vector = x[index]
        robust_distance = min_distance(vector, robust_vectors)
        boundary_distance = min_distance(vector, boundary_vectors)
        margin = boundary_distance - robust_distance
        pred_mean = float(vector @ weights["mean_delta"])
        pred_negative_rate = min(1.0, max(0.0, float(vector @ weights["negative_rate"])))
        pred_downside = max(0.0, float(vector @ weights["downside"]))
        pred_abs_delta = max(0.0, float(vector @ weights["abs_delta"]))
        tail_score = (
            pred_mean
            - 18.0 * pred_negative_rate
            - 0.35 * pred_downside
            - 0.08 * pred_abs_delta
            + 3.0 * margin
            - 0.75 * robust_distance
        )
        nearest_robust = min(
            robust_rows,
            key=lambda example: float(np.linalg.norm(vector - example["_vector"])),
        ) if robust_rows else None
        nearest_boundary = min(
            boundary_rows,
            key=lambda example: float(np.linalg.norm(vector - example["_vector"])),
        ) if boundary_rows else None
        row["tail_risk_score"] = tail_score
        row["predicted_seed_mean_delta_pp"] = pred_mean
        row["predicted_negative_seed_rate"] = pred_negative_rate
        row["predicted_downside_pp"] = pred_downside
        row["predicted_abs_seed_delta_pp"] = pred_abs_delta
        row["robust_distance"] = robust_distance
        row["boundary_distance"] = boundary_distance
        row["distance_margin"] = margin
        row["nearest_robust"] = {
            "match": nearest_robust["match"],
            "cut_minute": nearest_robust["cut_minute"],
            "lcb90_pp": nearest_robust["lcb90_pp"],
            "positive_seed_rate": nearest_robust["positive_seed_rate"],
        } if nearest_robust else None
        row["nearest_boundary"] = {
            "match": nearest_boundary["match"],
            "cut_minute": nearest_boundary["cut_minute"],
            "lcb90_pp": nearest_boundary["lcb90_pp"],
            "positive_seed_rate": nearest_boundary["positive_seed_rate"],
        } if nearest_boundary else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a seed-level tail-risk selector for intervention validation.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="HOME_SUB_ROLE_FORWARD")
    parser.add_argument("--default-cut-minute", type=int, default=45)
    parser.add_argument("--candidate-minutes", default="40,45,50,55,60,65")
    parser.add_argument("--max-candidate-matches", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=40.0)
    parser.add_argument("--min-lcb-pp", type=float, default=0.0)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.75)
    parser.add_argument("--min-seeds", type=int, default=4)
    parser.add_argument("--boundary-max-lcb-pp", type=float, default=0.0)
    parser.add_argument("--max-predicted-negative-rate", type=float, default=0.35)
    parser.add_argument("--min-predicted-mean-pp", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    label_paths = expand_inputs(args.labels)
    rows = [
        row
        for row in dedupe_rows(load_label_rows(label_paths))
        if action_name(row["action"]) == args.action
    ]
    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    label_rows = add_label_features(rows, event_by_name, args.default_cut_minute)
    if len(label_rows) < 8:
        raise RuntimeError(f"not enough labels: {len(label_rows)}")

    robust_rows = [
        row for row in label_rows
        if robust_label(row, args.min_lcb_pp, args.min_positive_seed_rate, args.min_seeds)
    ]
    boundary_rows = [row for row in label_rows if float(row["lcb90_pp"]) < args.boundary_max_lcb_pp]
    if not robust_rows or not boundary_rows:
        raise RuntimeError(f"need robust and boundary examples, got robust={len(robust_rows)} boundary={len(boundary_rows)}")

    names = feature_names(label_rows)
    x_label_raw = make_matrix(label_rows, names)
    x_label, mu, sigma = standardize(x_label_raw)
    for index, row in enumerate(label_rows):
        row["_vector"] = x_label[index]

    seeds = seed_rows(label_rows)
    x_seed = apply_standardize(make_matrix(seeds, names), mu, sigma)
    weights = {
        "mean_delta": fit_ridge(x_seed, np.asarray([row["seed_delta_pp"] for row in seeds]), args.ridge_alpha),
        "negative_rate": fit_ridge(x_seed, np.asarray([row["seed_negative"] for row in seeds]), args.ridge_alpha),
        "downside": fit_ridge(x_seed, np.asarray([row["seed_downside_pp"] for row in seeds]), args.ridge_alpha),
        "abs_delta": fit_ridge(x_seed, np.asarray([abs(row["seed_delta_pp"]) for row in seeds]), args.ridge_alpha),
    }

    seen = {(row["match"], f"ACTION::{row['action']}", row["cut_minute"]) for row in label_rows}
    candidates = make_candidates(
        args.checkpoint,
        event_by_name,
        args.action,
        parse_minutes(args.candidate_minutes),
        args.max_candidate_matches,
    )
    candidates = [row for row in candidates if (row["match"], row["action"], row["cut_minute"]) not in seen]
    score_candidates(candidates, names, mu, sigma, weights, robust_rows, boundary_rows)
    candidates = sorted(
        candidates,
        key=lambda row: (
            row["tail_risk_score"],
            -row["predicted_negative_seed_rate"],
            row["distance_margin"],
        ),
        reverse=True,
    )
    recommended = [
        row for row in candidates
        if row["tail_risk_score"] > 0.0
        and row["predicted_seed_mean_delta_pp"] >= args.min_predicted_mean_pp
        and row["predicted_negative_seed_rate"] <= args.max_predicted_negative_rate
        and row["distance_margin"] > 0.0
    ][: args.top_k]

    output = {
        "selector": "seed_tail_risk",
        "action": args.action,
        "label_rows": len(label_rows),
        "seed_rows": len(seeds),
        "robust_rows": len(robust_rows),
        "boundary_rows": len(boundary_rows),
        "thresholds": {
            "min_lcb_pp": args.min_lcb_pp,
            "min_positive_seed_rate": args.min_positive_seed_rate,
            "min_seeds": args.min_seeds,
            "boundary_max_lcb_pp": args.boundary_max_lcb_pp,
            "max_predicted_negative_rate": args.max_predicted_negative_rate,
            "min_predicted_mean_pp": args.min_predicted_mean_pp,
        },
        "training_summary": {
            "mean_seed_delta_pp": mean([row["seed_delta_pp"] for row in seeds]),
            "negative_seed_rate": mean([row["seed_negative"] for row in seeds]),
            "mean_downside_pp": mean([row["seed_downside_pp"] for row in seeds]),
            "robust_label_rate": len(robust_rows) / len(label_rows),
        },
        "recommended_validation": strip_vectors(recommended),
        "top_candidates": strip_vectors(candidates[: args.top_k]),
        "decision": "validate_recommended_candidates" if recommended else "no_tail_safe_candidates",
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "label_rows": output["label_rows"],
        "seed_rows": output["seed_rows"],
        "robust_rows": output["robust_rows"],
        "boundary_rows": output["boundary_rows"],
        "recommended_validation": len(output["recommended_validation"]),
        "decision": output["decision"],
    }, indent=2))
    for row in output["top_candidates"][:5]:
        print(
            f"{row['match']} minute={row['cut_minute']} "
            f"score={row['tail_risk_score']:.3f} mean={row['predicted_seed_mean_delta_pp']:.3f} "
            f"neg={row['predicted_negative_seed_rate']:.3f} margin={row['distance_margin']:.3f}"
        )


if __name__ == "__main__":
    main()
