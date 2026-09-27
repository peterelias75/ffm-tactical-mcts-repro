import argparse
import json
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
from .build_seed_tail_risk_selector import add_label_features
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


DEFAULT_LABELS = [
    "impactful_possession_pipeline/runs/rich_actions_3400matches_finetune_3epochs_20260712_maxlen2048/labels_forced_*.json",
]


def strong_rows(rows: Sequence[dict], min_lcb_pp: float, min_positive_seed_rate: float, min_seeds: int) -> list[dict]:
    return [
        row for row in rows
        if robust_label(row, min_lcb_pp, min_positive_seed_rate, min_seeds)
    ]


def bad_rows(rows: Sequence[dict], max_lcb_pp: float, max_positive_seed_rate: float) -> list[dict]:
    return [
        row for row in rows
        if float(row["lcb90_pp"]) < max_lcb_pp or float(row["positive_seed_rate"]) <= max_positive_seed_rate
    ]


def attach_vectors(rows: Sequence[dict], label_rows: Sequence[dict]) -> None:
    vectors = {
        (row["match"], row["action"], int(row["cut_minute"])): row["_vector"]
        for row in label_rows
    }
    for row in rows:
        row["_vector"] = vectors[(row["match"], row["action"], int(row["cut_minute"]))]


def score_candidates(candidates: list[dict], names: Sequence[str], mu: np.ndarray, sigma: np.ndarray, strong: Sequence[dict], bad: Sequence[dict]) -> None:
    x = apply_standardize(make_matrix(candidates, names), mu, sigma)
    strong_vectors = [row["_vector"] for row in strong]
    bad_vectors = [row["_vector"] for row in bad]
    for index, row in enumerate(candidates):
        vector = x[index]
        strong_distance = min_distance(vector, strong_vectors)
        bad_distance = min_distance(vector, bad_vectors)
        margin = bad_distance - strong_distance
        nearest_strong = min(strong, key=lambda example: float(np.linalg.norm(vector - example["_vector"])))
        nearest_bad = min(bad, key=lambda example: float(np.linalg.norm(vector - example["_vector"])))
        row["strong_neighbor_distance"] = strong_distance
        row["bad_neighbor_distance"] = bad_distance
        row["neighbor_margin"] = margin
        row["strong_neighbor_score"] = (
            5.0 * margin
            - 2.0 * strong_distance
            + 0.15 * float(nearest_strong["lcb90_pp"])
            + 4.0 * (float(nearest_strong["positive_seed_rate"]) - 0.75)
        )
        row["nearest_strong"] = {
            "match": nearest_strong["match"],
            "cut_minute": nearest_strong["cut_minute"],
            "lcb90_pp": nearest_strong["lcb90_pp"],
            "positive_seed_rate": nearest_strong["positive_seed_rate"],
        }
        row["nearest_bad"] = {
            "match": nearest_bad["match"],
            "cut_minute": nearest_bad["cut_minute"],
            "lcb90_pp": nearest_bad["lcb90_pp"],
            "positive_seed_rate": nearest_bad["positive_seed_rate"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample candidates nearest to confirmed strong forward-sub states.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="HOME_SUB_ROLE_FORWARD")
    parser.add_argument("--default-cut-minute", type=int, default=45)
    parser.add_argument("--candidate-minutes", default="50,55")
    parser.add_argument("--max-candidate-matches", type=int, default=1000)
    parser.add_argument("--min-strong-lcb-pp", type=float, default=5.0)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.75)
    parser.add_argument("--min-seeds", type=int, default=4)
    parser.add_argument("--bad-max-lcb-pp", type=float, default=0.0)
    parser.add_argument("--bad-max-positive-seed-rate", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--max-strong-distance", type=float, default=3.75)
    parser.add_argument("--min-score", type=float, default=-8.0)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    label_paths = expand_inputs(args.labels)
    raw_rows = [
        row for row in dedupe_rows(load_label_rows(label_paths))
        if action_name(row["action"]) == args.action
    ]
    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    label_rows = add_label_features(raw_rows, event_by_name, args.default_cut_minute)
    if len(label_rows) < 8:
        raise RuntimeError(f"not enough labels: {len(label_rows)}")

    names = feature_names(label_rows)
    x_label, mu, sigma = standardize(make_matrix(label_rows, names))
    for index, row in enumerate(label_rows):
        row["_vector"] = x_label[index]

    strong = strong_rows(label_rows, args.min_strong_lcb_pp, args.min_positive_seed_rate, args.min_seeds)
    bad = bad_rows(label_rows, args.bad_max_lcb_pp, args.bad_max_positive_seed_rate)
    if not strong or not bad:
        raise RuntimeError(f"need strong and bad examples, got strong={len(strong)} bad={len(bad)}")
    attach_vectors(strong, label_rows)
    attach_vectors(bad, label_rows)

    seen = {(row["match"], f"ACTION::{row['action']}", row["cut_minute"]) for row in label_rows}
    candidates = make_candidates(
        args.checkpoint,
        event_by_name,
        args.action,
        parse_minutes(args.candidate_minutes),
        args.max_candidate_matches,
    )
    candidates = [row for row in candidates if (row["match"], row["action"], row["cut_minute"]) not in seen]
    score_candidates(candidates, names, mu, sigma, strong, bad)
    candidates = sorted(
        candidates,
        key=lambda row: (
            row["strong_neighbor_score"],
            row["neighbor_margin"],
            -row["strong_neighbor_distance"],
        ),
        reverse=True,
    )
    recommended = [
        row for row in candidates
        if row["neighbor_margin"] >= args.min_margin
        and row["strong_neighbor_distance"] <= args.max_strong_distance
        and row["strong_neighbor_score"] >= args.min_score
    ][: args.top_k]

    output = {
        "selector": "strong_neighbor_sampler",
        "action": args.action,
        "label_rows": len(label_rows),
        "strong_rows": len(strong),
        "bad_rows": len(bad),
        "thresholds": {
            "candidate_minutes": args.candidate_minutes,
            "min_margin": args.min_margin,
            "max_strong_distance": args.max_strong_distance,
            "min_score": args.min_score,
        },
        "strong_examples": strip_vectors(strong),
        "recommended_validation": strip_vectors(recommended),
        "top_candidates": strip_vectors(candidates[: args.top_k]),
        "decision": "validate_recommended_candidates" if recommended else "no_strong_neighbor_candidates",
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "label_rows": output["label_rows"],
        "strong_rows": output["strong_rows"],
        "bad_rows": output["bad_rows"],
        "recommended_validation": len(output["recommended_validation"]),
        "decision": output["decision"],
    }, indent=2))
    for row in output["top_candidates"][:5]:
        print(
            f"{row['match']} minute={row['cut_minute']} score={row['strong_neighbor_score']:.3f} "
            f"margin={row['neighbor_margin']:.3f} dist={row['strong_neighbor_distance']:.3f} "
            f"bad_dist={row['bad_neighbor_distance']:.3f}"
        )


if __name__ == "__main__":
    main()
