import argparse
import json
from pathlib import Path
from typing import Sequence

from .build_corridor_aware_selector import (
    apply_standardize,
    feature_names,
    make_candidates,
    make_matrix,
    robust_label,
    standardize,
    strip_vectors,
)
from .build_seed_tail_risk_selector import add_label_features, fit_ridge, score_candidates, seed_rows
from .calibrate_tail_selector_thresholds import boundary_label
from .config import DataConfig
from .train_context_intervention_scorer import (
    action_name,
    checkpoint_event_map,
    dedupe_rows,
    expand_inputs,
    load_label_rows,
)
from .train_intervention_risk_gate import parse_minutes
from .utils import mean, save_json

import numpy as np


DEFAULT_LABELS = [
    "impactful_possession_pipeline/runs/rich_actions_3400matches_finetune_3epochs_20260712_maxlen2048/labels_forced_*.json",
]


def same_match_boundary_minutes(rows: Sequence[dict], boundary_max_lcb_pp: float, min_positive_seed_rate: float) -> dict[tuple[str, str], set[int]]:
    minutes: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        is_boundary = float(row["lcb90_pp"]) < boundary_max_lcb_pp or float(row["positive_seed_rate"]) < min_positive_seed_rate
        if not is_boundary:
            continue
        key = (row["match"], row["action"])
        minutes.setdefault(key, set()).add(int(row["cut_minute"]))
    return minutes


def interior_anchors(
    rows: Sequence[dict],
    min_lcb_pp: float,
    min_positive_seed_rate: float,
    min_seeds: int,
    boundary_max_lcb_pp: float,
    adjacent_minutes: int,
) -> list[dict]:
    boundary_minutes = same_match_boundary_minutes(rows, boundary_max_lcb_pp, min_positive_seed_rate)
    strong = [
        row for row in rows
        if robust_label(row, min_lcb_pp, min_positive_seed_rate, min_seeds)
    ]
    anchors = []
    for row in strong:
        near_boundary = any(
            abs(int(row["cut_minute"]) - minute) <= adjacent_minutes
            for minute in boundary_minutes.get((row["match"], row["action"]), set())
        )
        item = dict(row)
        item["interior_anchor"] = not near_boundary
        anchors.append(item)
    interior = [row for row in anchors if row["interior_anchor"]]
    return interior if interior else anchors


def fit_tail_weights(label_rows: Sequence[dict], names: Sequence[str], mu: np.ndarray, sigma: np.ndarray, ridge_alpha: float) -> dict[str, np.ndarray]:
    seeds = seed_rows(label_rows)
    x_seed = apply_standardize(make_matrix(seeds, names), mu, sigma)
    return {
        "mean_delta": fit_ridge(x_seed, np.asarray([row["seed_delta_pp"] for row in seeds]), ridge_alpha),
        "negative_rate": fit_ridge(x_seed, np.asarray([row["seed_negative"] for row in seeds]), ridge_alpha),
        "downside": fit_ridge(x_seed, np.asarray([row["seed_downside_pp"] for row in seeds]), ridge_alpha),
        "abs_delta": fit_ridge(x_seed, np.asarray([abs(row["seed_delta_pp"]) for row in seeds]), ridge_alpha),
    }


def attach_label_vectors(rows: Sequence[dict], label_rows: Sequence[dict]) -> None:
    vectors = {
        (row["match"], row["action"], int(row["cut_minute"])): row["_vector"]
        for row in label_rows
        if "_vector" in row
    }
    for row in rows:
        key = (row["match"], row["action"], int(row["cut_minute"]))
        row["_vector"] = vectors[key]


def score_interior_candidates(candidates: list[dict], anchors: Sequence[dict]) -> None:
    for row in candidates:
        nearest = row.get("nearest_robust") or {}
        anchor_bonus = 0.0
        for anchor in anchors:
            if anchor["match"] == nearest.get("match") and int(anchor["cut_minute"]) == int(nearest.get("cut_minute", -999)):
                anchor_bonus = 1.0
                break
        row["interior_corridor_score"] = (
            row["tail_risk_score"]
            + 6.0 * row["distance_margin"]
            - 1.25 * row["robust_distance"]
            - 20.0 * row["predicted_negative_seed_rate"]
            + 0.20 * row["predicted_seed_mean_delta_pp"]
            + 4.0 * anchor_bonus
        )
        row["nearest_anchor_is_interior"] = bool(anchor_bonus)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates by similarity to strong corridor interiors.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="HOME_SUB_ROLE_FORWARD")
    parser.add_argument("--default-cut-minute", type=int, default=45)
    parser.add_argument("--candidate-minutes", default="30,35,40,45,50,55,60,65,70,75")
    parser.add_argument("--max-candidate-matches", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=40.0)
    parser.add_argument("--min-anchor-lcb-pp", type=float, default=5.0)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.75)
    parser.add_argument("--min-seeds", type=int, default=4)
    parser.add_argument("--boundary-max-lcb-pp", type=float, default=5.0)
    parser.add_argument("--adjacent-boundary-minutes", type=int, default=5)
    parser.add_argument("--max-predicted-negative-rate", type=float, default=0.30)
    parser.add_argument("--min-predicted-mean-pp", type=float, default=8.0)
    parser.add_argument("--min-distance-margin", type=float, default=0.75)
    parser.add_argument("--max-anchor-distance", type=float, default=3.5)
    parser.add_argument("--min-interior-score", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    label_paths = expand_inputs(args.labels)
    rows = [
        row for row in dedupe_rows(load_label_rows(label_paths))
        if action_name(row["action"]) == args.action
    ]
    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    label_rows = add_label_features(rows, event_by_name, args.default_cut_minute)
    if len(label_rows) < 8:
        raise RuntimeError(f"not enough labels: {len(label_rows)}")

    anchors = interior_anchors(
        label_rows,
        args.min_anchor_lcb_pp,
        args.min_positive_seed_rate,
        args.min_seeds,
        args.boundary_max_lcb_pp,
        args.adjacent_boundary_minutes,
    )
    boundary_rows = [
        row for row in label_rows
        if boundary_label(row, args.boundary_max_lcb_pp) or float(row["positive_seed_rate"]) < args.min_positive_seed_rate
    ]
    if not anchors or not boundary_rows:
        raise RuntimeError(f"need anchors and boundaries, got anchors={len(anchors)} boundary={len(boundary_rows)}")

    names = feature_names(label_rows)
    x_label, mu, sigma = standardize(make_matrix(label_rows, names))
    for index, row in enumerate(label_rows):
        row["_vector"] = x_label[index]
    attach_label_vectors(anchors, label_rows)
    attach_label_vectors(boundary_rows, label_rows)

    weights = fit_tail_weights(label_rows, names, mu, sigma, args.ridge_alpha)
    seen = {(row["match"], f"ACTION::{row['action']}", row["cut_minute"]) for row in label_rows}
    candidates = make_candidates(
        args.checkpoint,
        event_by_name,
        args.action,
        parse_minutes(args.candidate_minutes),
        args.max_candidate_matches,
    )
    candidates = [row for row in candidates if (row["match"], row["action"], row["cut_minute"]) not in seen]
    score_candidates(candidates, names, mu, sigma, weights, anchors, boundary_rows)
    score_interior_candidates(candidates, anchors)
    candidates = sorted(
        candidates,
        key=lambda row: (
            row["interior_corridor_score"],
            row["distance_margin"],
            -row["predicted_negative_seed_rate"],
            row["predicted_seed_mean_delta_pp"],
        ),
        reverse=True,
    )
    recommended = [
        row for row in candidates
        if row["nearest_anchor_is_interior"]
        and row["interior_corridor_score"] >= args.min_interior_score
        and row["predicted_negative_seed_rate"] <= args.max_predicted_negative_rate
        and row["predicted_seed_mean_delta_pp"] >= args.min_predicted_mean_pp
        and row["distance_margin"] >= args.min_distance_margin
        and row["robust_distance"] <= args.max_anchor_distance
    ][: args.top_k]

    output = {
        "selector": "interior_corridor_similarity",
        "action": args.action,
        "label_rows": len(label_rows),
        "anchor_rows": len(anchors),
        "boundary_rows": len(boundary_rows),
        "thresholds": {
            "min_anchor_lcb_pp": args.min_anchor_lcb_pp,
            "min_positive_seed_rate": args.min_positive_seed_rate,
            "min_seeds": args.min_seeds,
            "boundary_max_lcb_pp": args.boundary_max_lcb_pp,
            "adjacent_boundary_minutes": args.adjacent_boundary_minutes,
            "max_predicted_negative_rate": args.max_predicted_negative_rate,
            "min_predicted_mean_pp": args.min_predicted_mean_pp,
            "min_distance_margin": args.min_distance_margin,
            "max_anchor_distance": args.max_anchor_distance,
            "min_interior_score": args.min_interior_score,
        },
        "anchors": strip_vectors(anchors),
        "recommended_validation": strip_vectors(recommended),
        "top_candidates": strip_vectors(candidates[: args.top_k]),
        "decision": "validate_recommended_candidates" if recommended else "no_interior_corridor_candidates",
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "label_rows": output["label_rows"],
        "anchor_rows": output["anchor_rows"],
        "boundary_rows": output["boundary_rows"],
        "recommended_validation": len(output["recommended_validation"]),
        "decision": output["decision"],
    }, indent=2))
    for row in output["top_candidates"][:5]:
        print(
            f"{row['match']} minute={row['cut_minute']} score={row['interior_corridor_score']:.3f} "
            f"neg={row['predicted_negative_seed_rate']:.3f} mean={row['predicted_seed_mean_delta_pp']:.3f} "
            f"margin={row['distance_margin']:.3f} dist={row['robust_distance']:.3f}"
        )


if __name__ == "__main__":
    main()
