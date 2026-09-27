import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .build_corridor_aware_selector import (
    apply_standardize,
    feature_names,
    make_candidates,
    make_matrix,
    robust_label,
    standardize,
    strip_vectors,
)
from .build_seed_tail_risk_selector import (
    add_label_features,
    fit_ridge,
    score_candidates,
    seed_rows,
)
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


def boundary_label(row: dict, max_lcb_pp: float) -> bool:
    return float(row["lcb90_pp"]) < max_lcb_pp


def train_and_score_rows(
    train_rows: Sequence[dict],
    test_rows: Sequence[dict],
    ridge_alpha: float,
    min_lcb_pp: float,
    min_positive_seed_rate: float,
    min_seeds: int,
    boundary_max_lcb_pp: float,
) -> list[dict]:
    robust_rows = [
        row for row in train_rows
        if robust_label(row, min_lcb_pp, min_positive_seed_rate, min_seeds)
    ]
    boundary_rows = [row for row in train_rows if boundary_label(row, boundary_max_lcb_pp)]
    seeds = seed_rows(train_rows)
    if len(train_rows) < 8 or len(seeds) < 12 or not robust_rows or not boundary_rows:
        return []

    names = feature_names(train_rows)
    x_label, mu, sigma = standardize(make_matrix(train_rows, names))
    for index, row in enumerate(train_rows):
        row["_vector"] = x_label[index]

    x_seed = apply_standardize(make_matrix(seeds, names), mu, sigma)
    weights = {
        "mean_delta": fit_ridge(x_seed, np.asarray([row["seed_delta_pp"] for row in seeds]), ridge_alpha),
        "negative_rate": fit_ridge(x_seed, np.asarray([row["seed_negative"] for row in seeds]), ridge_alpha),
        "downside": fit_ridge(x_seed, np.asarray([row["seed_downside_pp"] for row in seeds]), ridge_alpha),
        "abs_delta": fit_ridge(x_seed, np.asarray([abs(row["seed_delta_pp"]) for row in seeds]), ridge_alpha),
    }

    scored = [dict(row) for row in test_rows]
    score_candidates(scored, names, mu, sigma, weights, robust_rows, boundary_rows)
    return scored


def leave_match_out_scores(
    label_rows: Sequence[dict],
    ridge_alpha: float,
    min_lcb_pp: float,
    min_positive_seed_rate: float,
    min_seeds: int,
    boundary_max_lcb_pp: float,
) -> list[dict]:
    by_match: dict[str, list[dict]] = defaultdict(list)
    for row in label_rows:
        by_match[row["match"]].append(row)

    scored = []
    for match, heldout_rows in sorted(by_match.items()):
        train_rows = [dict(row) for row in label_rows if row["match"] != match]
        group_scores = train_and_score_rows(
            train_rows,
            heldout_rows,
            ridge_alpha,
            min_lcb_pp,
            min_positive_seed_rate,
            min_seeds,
            boundary_max_lcb_pp,
        )
        for row in group_scores:
            row["holdout_match"] = match
            row["true_robust"] = robust_label(row, min_lcb_pp, min_positive_seed_rate, min_seeds)
            row["true_strong"] = robust_label(row, 5.0, min_positive_seed_rate, min_seeds)
            row["true_failure"] = not row["true_robust"]
        scored.extend(group_scores)
    return scored


def summarize_selection(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "coverage": 0,
            "true_robust_rate": 0.0,
            "true_strong_rate": 0.0,
            "false_positive_count": 0,
            "worst_lcb90_pp": None,
            "mean_lcb90_pp": None,
            "mean_rollout_delta_pp": None,
            "mean_true_positive_seed_rate": None,
        }
    return {
        "coverage": len(rows),
        "true_robust_rate": mean([1.0 if row["true_robust"] else 0.0 for row in rows]),
        "true_strong_rate": mean([1.0 if row["true_strong"] else 0.0 for row in rows]),
        "false_positive_count": sum(1 for row in rows if row["true_failure"]),
        "worst_lcb90_pp": min(float(row["lcb90_pp"]) for row in rows),
        "mean_lcb90_pp": mean([float(row["lcb90_pp"]) for row in rows]),
        "mean_rollout_delta_pp": mean([float(row["mean_rollout_delta_pp"]) for row in rows]),
        "mean_true_positive_seed_rate": mean([float(row["positive_seed_rate"]) for row in rows]),
    }


def threshold_grid(scored_rows: Sequence[dict], min_coverage: int) -> list[dict]:
    configs = []
    for min_score in [-6.0, -3.0, 0.0, 2.0, 4.0, 6.0, 8.0]:
        for max_neg in [0.25, 0.30, 0.35, 0.40, 0.45]:
            for min_mean in [0.0, 5.0, 10.0, 15.0]:
                for min_margin in [-1.0, 0.0, 0.25, 0.50, 1.0]:
                    for max_robust_distance in [2.5, 3.0, 3.5, 4.0, 5.0, 999.0]:
                        selected = [
                            row for row in scored_rows
                            if float(row["tail_risk_score"]) >= min_score
                            and float(row["predicted_negative_seed_rate"]) <= max_neg
                            and float(row["predicted_seed_mean_delta_pp"]) >= min_mean
                            and float(row["distance_margin"]) >= min_margin
                            and float(row["robust_distance"]) <= max_robust_distance
                        ]
                        if len(selected) < min_coverage:
                            continue
                        summary = summarize_selection(selected)
                        configs.append({
                            "thresholds": {
                                "min_tail_risk_score": min_score,
                                "max_predicted_negative_seed_rate": max_neg,
                                "min_predicted_seed_mean_delta_pp": min_mean,
                                "min_distance_margin": min_margin,
                                "max_robust_distance": None if max_robust_distance >= 999.0 else max_robust_distance,
                            },
                            **summary,
                            "selected": strip_vectors(selected),
                        })
    return sorted(
        configs,
        key=lambda row: (
            row["false_positive_count"] > 0,
            row["false_positive_count"],
            -(row["worst_lcb90_pp"] if row["worst_lcb90_pp"] is not None else -999.0),
            -row["true_strong_rate"],
            -row["true_robust_rate"],
            -row["coverage"],
            -(row["mean_lcb90_pp"] if row["mean_lcb90_pp"] is not None else -999.0),
        ),
    )


def current_candidates(
    label_rows: Sequence[dict],
    checkpoint: str,
    event_by_name: dict[str, str],
    action: str,
    candidate_minutes: str,
    max_candidate_matches: int,
    ridge_alpha: float,
    min_lcb_pp: float,
    min_positive_seed_rate: float,
    min_seeds: int,
    boundary_max_lcb_pp: float,
) -> list[dict]:
    seen = {(row["match"], f"ACTION::{row['action']}", row["cut_minute"]) for row in label_rows}
    candidates = make_candidates(
        checkpoint,
        event_by_name,
        action,
        parse_minutes(candidate_minutes),
        max_candidate_matches,
    )
    candidates = [row for row in candidates if (row["match"], row["action"], row["cut_minute"]) not in seen]
    return train_and_score_rows(
        [dict(row) for row in label_rows],
        candidates,
        ridge_alpha,
        min_lcb_pp,
        min_positive_seed_rate,
        min_seeds,
        boundary_max_lcb_pp,
    )


def select_with_thresholds(rows: Sequence[dict], thresholds: dict, top_k: int) -> list[dict]:
    max_robust_distance = thresholds.get("max_robust_distance")
    selected = [
        row for row in rows
        if float(row["tail_risk_score"]) >= thresholds["min_tail_risk_score"]
        and float(row["predicted_negative_seed_rate"]) <= thresholds["max_predicted_negative_seed_rate"]
        and float(row["predicted_seed_mean_delta_pp"]) >= thresholds["min_predicted_seed_mean_delta_pp"]
        and float(row["distance_margin"]) >= thresholds["min_distance_margin"]
        and (max_robust_distance is None or float(row["robust_distance"]) <= max_robust_distance)
    ]
    return sorted(
        selected,
        key=lambda row: (
            row["tail_risk_score"],
            -row["predicted_negative_seed_rate"],
            row["predicted_seed_mean_delta_pp"],
        ),
        reverse=True,
    )[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate seed-tail selector thresholds on held-out rollout labels.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="HOME_SUB_ROLE_FORWARD")
    parser.add_argument("--default-cut-minute", type=int, default=45)
    parser.add_argument("--candidate-minutes", default="30,35,40,45,50,55,60,65,70,75")
    parser.add_argument("--max-candidate-matches", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=40.0)
    parser.add_argument("--min-lcb-pp", type=float, default=0.0)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.75)
    parser.add_argument("--min-seeds", type=int, default=4)
    parser.add_argument("--boundary-max-lcb-pp", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=int, default=2)
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

    scored = leave_match_out_scores(
        label_rows,
        args.ridge_alpha,
        args.min_lcb_pp,
        args.min_positive_seed_rate,
        args.min_seeds,
        args.boundary_max_lcb_pp,
    )
    configs = threshold_grid(scored, args.min_coverage)
    best_config = configs[0] if configs else None
    candidate_rows = current_candidates(
        label_rows,
        args.checkpoint,
        event_by_name,
        args.action,
        args.candidate_minutes,
        args.max_candidate_matches,
        args.ridge_alpha,
        args.min_lcb_pp,
        args.min_positive_seed_rate,
        args.min_seeds,
        args.boundary_max_lcb_pp,
    )
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            row["tail_risk_score"],
            -row["predicted_negative_seed_rate"],
            row["predicted_seed_mean_delta_pp"],
        ),
        reverse=True,
    )
    best_config_is_safe = bool(
        best_config
        and best_config["false_positive_count"] == 0
        and best_config["true_robust_rate"] >= 1.0
    )
    recommended = (
        select_with_thresholds(candidate_rows, best_config["thresholds"], args.top_k)
        if best_config_is_safe
        else []
    )

    output = {
        "calibrator": "leave_match_out_seed_tail_thresholds",
        "action": args.action,
        "label_inputs": [str(path) for path in label_paths],
        "label_rows": len(label_rows),
        "holdout_scored_rows": len(scored),
        "true_robust_rows": sum(
            1 for row in label_rows
            if robust_label(row, args.min_lcb_pp, args.min_positive_seed_rate, args.min_seeds)
        ),
        "true_strong_rows": sum(
            1 for row in label_rows
            if robust_label(row, 5.0, args.min_positive_seed_rate, args.min_seeds)
        ),
        "configs_tested_with_coverage": len(configs),
        "best_configs": [
            {key: value for key, value in config.items() if key != "selected"}
            for config in configs[:20]
        ],
        "best_config_selected_holdouts": best_config["selected"] if best_config else [],
        "recommended_current_validation": strip_vectors(recommended),
        "top_current_candidates": strip_vectors(candidate_rows[: args.top_k]),
        "decision": "validate_recommended_candidates" if recommended else "no_calibrated_safe_candidates",
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "label_rows": output["label_rows"],
        "holdout_scored_rows": output["holdout_scored_rows"],
        "true_robust_rows": output["true_robust_rows"],
        "true_strong_rows": output["true_strong_rows"],
        "configs_tested_with_coverage": output["configs_tested_with_coverage"],
        "recommended_current_validation": len(output["recommended_current_validation"]),
        "decision": output["decision"],
    }, indent=2))
    for config in output["best_configs"][:5]:
        print(
            f"coverage={config['coverage']} robust={config['true_robust_rate']:.3f} "
            f"strong={config['true_strong_rate']:.3f} fp={config['false_positive_count']} "
            f"worst_lcb={config['worst_lcb90_pp']}"
        )
    for row in output["recommended_current_validation"][:5]:
        print(
            f"candidate {row['match']} minute={row['cut_minute']} "
            f"score={row['tail_risk_score']:.3f} neg={row['predicted_negative_seed_rate']:.3f} "
            f"mean={row['predicted_seed_mean_delta_pp']:.3f}"
        )


if __name__ == "__main__":
    main()
