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
    load_label_rows,
)
from .train_intervention_risk_gate import checkpoint_test_files, parse_minutes
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


def feature_names(rows: Sequence[dict]) -> list[str]:
    names = set()
    for row in rows:
        names.update(row["features"])
    excluded = {"value_head_delta_pp", "value_head_delta_sq", "noop_home_possession"}
    return ["bias"] + sorted(name for name in names if name != "bias" and name not in excluded)


def make_matrix(rows: Sequence[dict], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in names] for row in rows], dtype=np.float64)


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    mu[0] = 0.0
    sigma[0] = 1.0
    return (x - mu) / sigma, mu, sigma


def apply_standardize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


def min_distance(vector: np.ndarray, examples: Sequence[np.ndarray]) -> float:
    if not examples:
        return 0.0
    return min(float(np.linalg.norm(vector - example)) for example in examples)


def add_features(rows: Sequence[dict], event_by_name: dict[str, str], default_cut_minute: int) -> list[dict]:
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


def robust_label(row: dict, min_lcb_pp: float, min_positive_seed_rate: float, min_seeds: int) -> bool:
    return (
        int(row["n_seeds"]) >= min_seeds
        and float(row["lcb90_pp"]) >= min_lcb_pp
        and float(row["positive_seed_rate"]) >= min_positive_seed_rate
    )


def boundary_label(row: dict, max_lcb_pp: float) -> bool:
    return float(row["lcb90_pp"]) < max_lcb_pp


def make_candidates(
    checkpoint: str,
    event_by_name: dict[str, str],
    action: str,
    minutes: Sequence[int],
    max_matches: int,
) -> list[dict]:
    candidates = []
    base_row = {
        "value_head_delta_pp": 0.0,
        "mean_noop_home_possession": 0.5,
        "action": action,
    }
    for match_json in checkpoint_test_files(checkpoint)[:max_matches]:
        event_file = event_by_name.get(Path(match_json).name, match_json)
        if not Path(event_file).exists():
            continue
        events = load_json(event_file)
        for minute in minutes:
            candidates.append(
                {
                    "match": Path(event_file).name,
                    "match_json": event_file,
                    "action": f"ACTION::{action}",
                    "cut_minute": int(minute),
                    "end_minute": int(minute) + 5,
                    "features": context_features(events, action, base_row, int(minute)),
                }
            )
    return candidates


def score_candidates(
    candidates: list[dict],
    names: Sequence[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    robust_vectors: Sequence[np.ndarray],
    boundary_vectors: Sequence[np.ndarray],
    robust_rows: Sequence[dict],
    boundary_rows: Sequence[dict],
) -> None:
    x = apply_standardize(make_matrix(candidates, names), mu, sigma)
    robust_lcb = mean([float(row["lcb90_pp"]) for row in robust_rows])
    robust_pos = mean([float(row["positive_seed_rate"]) for row in robust_rows])
    for index, row in enumerate(candidates):
        vector = x[index]
        robust_distance = min_distance(vector, robust_vectors)
        boundary_distance = min_distance(vector, boundary_vectors)
        margin = boundary_distance - robust_distance
        nearest_robust = min(
            robust_rows,
            key=lambda example: float(np.linalg.norm(vector - example["_vector"])),
        ) if robust_rows else None
        nearest_boundary = min(
            boundary_rows,
            key=lambda example: float(np.linalg.norm(vector - example["_vector"])),
        ) if boundary_rows else None
        row["corridor_score"] = (
            4.0 * margin
            - 1.5 * robust_distance
            + 0.08 * robust_lcb
            + 4.0 * (robust_pos - 0.75)
        )
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


def strip_vectors(rows: Sequence[dict]) -> list[dict]:
    clean = []
    for row in rows:
        item = dict(row)
        item.pop("features", None)
        item.pop("_vector", None)
        clean.append(item)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank intervention states by similarity to confirmed local corridors.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="HOME_SUB_ROLE_FORWARD")
    parser.add_argument("--default-cut-minute", type=int, default=45)
    parser.add_argument("--candidate-minutes", default="40,45,50,55,60,65")
    parser.add_argument("--max-candidate-matches", type=int, default=1000)
    parser.add_argument("--min-lcb-pp", type=float, default=0.0)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.75)
    parser.add_argument("--min-seeds", type=int, default=4)
    parser.add_argument("--boundary-max-lcb-pp", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    label_paths = expand_inputs(args.labels)
    rows = [
        row
        for row in dedupe_rows(load_label_rows(label_paths))
        if action_name(row["action"]) == args.action
    ]
    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    label_rows = add_features(rows, event_by_name, args.default_cut_minute)
    if len(label_rows) < 8:
        raise RuntimeError(f"not enough labels: {len(label_rows)}")

    robust_rows = [
        row for row in label_rows
        if robust_label(row, args.min_lcb_pp, args.min_positive_seed_rate, args.min_seeds)
    ]
    boundary_rows = [row for row in label_rows if boundary_label(row, args.boundary_max_lcb_pp)]
    if not robust_rows or not boundary_rows:
        raise RuntimeError(f"need robust and boundary examples, got robust={len(robust_rows)} boundary={len(boundary_rows)}")

    names = feature_names(label_rows)
    x_raw = make_matrix(label_rows, names)
    x, mu, sigma = standardize(x_raw)
    for index, row in enumerate(label_rows):
        row["_vector"] = x[index]
    robust_vectors = [row["_vector"] for row in robust_rows]
    boundary_vectors = [row["_vector"] for row in boundary_rows]

    seen = {(row["match"], f"ACTION::{row['action']}", row["cut_minute"]) for row in label_rows}
    candidates = make_candidates(
        args.checkpoint,
        event_by_name,
        args.action,
        parse_minutes(args.candidate_minutes),
        args.max_candidate_matches,
    )
    candidates = [row for row in candidates if (row["match"], row["action"], row["cut_minute"]) not in seen]
    score_candidates(candidates, names, mu, sigma, robust_vectors, boundary_vectors, robust_rows, boundary_rows)
    candidates = sorted(
        candidates,
        key=lambda row: (
            row["corridor_score"],
            row["distance_margin"],
            -row["robust_distance"],
        ),
        reverse=True,
    )

    recommended = [
        row for row in candidates
        if row["distance_margin"] > 0.0 and row["corridor_score"] > 0.0
    ][: args.top_k]
    output = {
        "selector": "corridor_aware_similarity",
        "action": args.action,
        "label_rows": len(label_rows),
        "robust_rows": len(robust_rows),
        "boundary_rows": len(boundary_rows),
        "thresholds": {
            "min_lcb_pp": args.min_lcb_pp,
            "min_positive_seed_rate": args.min_positive_seed_rate,
            "min_seeds": args.min_seeds,
            "boundary_max_lcb_pp": args.boundary_max_lcb_pp,
        },
        "robust_examples": strip_vectors(robust_rows),
        "boundary_examples": strip_vectors(boundary_rows),
        "recommended_validation": strip_vectors(recommended),
        "top_candidates": strip_vectors(candidates[: args.top_k]),
        "decision": "validate_recommended_candidates" if recommended else "no_corridor_similarity_candidates",
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "label_rows": output["label_rows"],
        "robust_rows": output["robust_rows"],
        "boundary_rows": output["boundary_rows"],
        "recommended_validation": len(output["recommended_validation"]),
        "decision": output["decision"],
    }, indent=2))
    for row in output["top_candidates"][:5]:
        print(
            f"{row['match']} minute={row['cut_minute']} "
            f"score={row['corridor_score']:.3f} margin={row['distance_margin']:.3f} "
            f"robust_dist={row['robust_distance']:.3f}"
        )


if __name__ == "__main__":
    main()
