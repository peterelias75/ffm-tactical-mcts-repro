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


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def label_target(row: dict, strong_lcb_pp: float, min_positive_seed_rate: float, min_seeds: int) -> float:
    return 1.0 if robust_label(row, strong_lcb_pp, min_positive_seed_rate, min_seeds) else 0.0


def downside_target(row: dict) -> float:
    return max(0.0, -min(float(value) for value in row["seed_deltas_pp"]))


def train_models(rows: Sequence[dict], ridge_alpha: float, strong_lcb_pp: float, min_positive_seed_rate: float, min_seeds: int) -> dict:
    names = feature_names(rows)
    x, mu, sigma = standardize(make_matrix(rows, names))
    y_robust = np.asarray([label_target(row, strong_lcb_pp, min_positive_seed_rate, min_seeds) for row in rows])
    y_lcb = np.asarray([float(row["lcb90_pp"]) for row in rows])
    y_pos = np.asarray([float(row["positive_seed_rate"]) for row in rows])
    y_downside = np.asarray([downside_target(row) for row in rows])
    return {
        "feature_names": names,
        "mu": mu,
        "sigma": sigma,
        "weights": {
            "robust_probability": fit_ridge(x, y_robust, ridge_alpha),
            "lcb90_pp": fit_ridge(x, y_lcb, ridge_alpha),
            "positive_seed_rate": fit_ridge(x, y_pos, ridge_alpha),
            "downside_pp": fit_ridge(x, y_downside, ridge_alpha),
        },
    }


def score_rows(rows: list[dict], model: dict) -> None:
    x = apply_standardize(make_matrix(rows, model["feature_names"]), model["mu"], model["sigma"])
    weights = model["weights"]
    for index, row in enumerate(rows):
        robust_probability = float(x[index] @ weights["robust_probability"])
        predicted_lcb = float(x[index] @ weights["lcb90_pp"])
        predicted_pos = float(x[index] @ weights["positive_seed_rate"])
        predicted_downside = float(x[index] @ weights["downside_pp"])
        row["predicted_robust_probability"] = max(0.0, min(1.0, robust_probability))
        row["predicted_lcb90_pp"] = predicted_lcb
        row["predicted_positive_seed_rate"] = max(0.0, min(1.0, predicted_pos))
        row["predicted_downside_pp"] = max(0.0, predicted_downside)
        row["robust_boundary_score"] = (
            20.0 * row["predicted_robust_probability"]
            + 0.8 * row["predicted_lcb90_pp"]
            + 8.0 * (row["predicted_positive_seed_rate"] - 0.75)
            - 0.35 * row["predicted_downside_pp"]
        )


def leave_match_out_scores(rows: Sequence[dict], ridge_alpha: float, strong_lcb_pp: float, min_positive_seed_rate: float, min_seeds: int) -> list[dict]:
    by_match: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_match[row["match"]].append(row)
    scored = []
    for match, heldout in sorted(by_match.items()):
        train_rows = [row for row in rows if row["match"] != match]
        if len(train_rows) < 8:
            continue
        model = train_models(train_rows, ridge_alpha, strong_lcb_pp, min_positive_seed_rate, min_seeds)
        test_rows = [dict(row) for row in heldout]
        score_rows(test_rows, model)
        for row in test_rows:
            row["true_strong"] = bool(label_target(row, strong_lcb_pp, min_positive_seed_rate, min_seeds))
            row["true_robust"] = bool(label_target(row, 0.0, min_positive_seed_rate, min_seeds))
            row["true_failure"] = not row["true_robust"]
        scored.extend(test_rows)
    return scored


def summarize(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "coverage": 0,
            "true_robust_rate": 0.0,
            "true_strong_rate": 0.0,
            "false_positive_count": 0,
            "worst_lcb90_pp": None,
            "mean_lcb90_pp": None,
            "mean_rollout_delta_pp": None,
        }
    return {
        "coverage": len(rows),
        "true_robust_rate": mean([1.0 if row["true_robust"] else 0.0 for row in rows]),
        "true_strong_rate": mean([1.0 if row["true_strong"] else 0.0 for row in rows]),
        "false_positive_count": sum(1 for row in rows if row["true_failure"]),
        "worst_lcb90_pp": min(float(row["lcb90_pp"]) for row in rows),
        "mean_lcb90_pp": mean([float(row["lcb90_pp"]) for row in rows]),
        "mean_rollout_delta_pp": mean([float(row["mean_rollout_delta_pp"]) for row in rows]),
    }


def grid_thresholds(scored_rows: Sequence[dict], min_coverage: int) -> list[dict]:
    configs = []
    for min_prob in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        for min_lcb in [-5.0, 0.0, 2.5, 5.0, 8.0]:
            for min_pos in [0.60, 0.70, 0.75, 0.80]:
                for max_downside in [20.0, 30.0, 40.0, 60.0, 999.0]:
                    selected = [
                        row for row in scored_rows
                        if float(row["predicted_robust_probability"]) >= min_prob
                        and float(row["predicted_lcb90_pp"]) >= min_lcb
                        and float(row["predicted_positive_seed_rate"]) >= min_pos
                        and float(row["predicted_downside_pp"]) <= max_downside
                    ]
                    if len(selected) < min_coverage:
                        continue
                    configs.append({
                        "thresholds": {
                            "min_predicted_robust_probability": min_prob,
                            "min_predicted_lcb90_pp": min_lcb,
                            "min_predicted_positive_seed_rate": min_pos,
                            "max_predicted_downside_pp": None if max_downside >= 999.0 else max_downside,
                        },
                        **summarize(selected),
                        "selected": strip_vectors(selected),
                    })
    return sorted(
        configs,
        key=lambda row: (
            row["false_positive_count"] > 0,
            row["false_positive_count"],
            -(row["worst_lcb90_pp"] if row["worst_lcb90_pp"] is not None else -999.0),
            -row["true_robust_rate"],
            -row["true_strong_rate"],
            -row["coverage"],
        ),
    )


def passes(row: dict, thresholds: dict) -> bool:
    max_downside = thresholds.get("max_predicted_downside_pp")
    return (
        float(row["predicted_robust_probability"]) >= thresholds["min_predicted_robust_probability"]
        and float(row["predicted_lcb90_pp"]) >= thresholds["min_predicted_lcb90_pp"]
        and float(row["predicted_positive_seed_rate"]) >= thresholds["min_predicted_positive_seed_rate"]
        and (max_downside is None or float(row["predicted_downside_pp"]) <= max_downside)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a robust-vs-boundary classifier for forward intervention candidates.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="HOME_SUB_ROLE_FORWARD")
    parser.add_argument("--default-cut-minute", type=int, default=45)
    parser.add_argument("--candidate-minutes", default="30,35,40,45,50,55,60,65,70,75")
    parser.add_argument("--max-candidate-matches", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--strong-lcb-pp", type=float, default=5.0)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.75)
    parser.add_argument("--min-seeds", type=int, default=4)
    parser.add_argument("--min-coverage", type=int, default=2)
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

    holdout_rows = leave_match_out_scores(
        label_rows,
        args.ridge_alpha,
        args.strong_lcb_pp,
        args.min_positive_seed_rate,
        args.min_seeds,
    )
    configs = grid_thresholds(holdout_rows, args.min_coverage)
    best_config = configs[0] if configs else None
    best_config_is_safe = bool(
        best_config
        and best_config["false_positive_count"] == 0
        and best_config["true_robust_rate"] >= 1.0
    )

    model = train_models(label_rows, args.ridge_alpha, args.strong_lcb_pp, args.min_positive_seed_rate, args.min_seeds)
    seen = {(row["match"], f"ACTION::{row['action']}", row["cut_minute"]) for row in label_rows}
    candidates = make_candidates(
        args.checkpoint,
        event_by_name,
        args.action,
        parse_minutes(args.candidate_minutes),
        args.max_candidate_matches,
    )
    candidates = [row for row in candidates if (row["match"], row["action"], row["cut_minute"]) not in seen]
    score_rows(candidates, model)
    candidates = sorted(
        candidates,
        key=lambda row: (
            row["robust_boundary_score"],
            row["predicted_robust_probability"],
            row["predicted_lcb90_pp"],
        ),
        reverse=True,
    )
    recommended = [
        row for row in candidates
        if best_config_is_safe and passes(row, best_config["thresholds"])
    ][: args.top_k]

    output = {
        "selector": "robust_boundary_classifier",
        "action": args.action,
        "label_rows": len(label_rows),
        "holdout_scored_rows": len(holdout_rows),
        "true_strong_rows": sum(
            1 for row in label_rows
            if label_target(row, args.strong_lcb_pp, args.min_positive_seed_rate, args.min_seeds)
        ),
        "configs_tested_with_coverage": len(configs),
        "best_configs": [
            {key: value for key, value in config.items() if key != "selected"}
            for config in configs[:20]
        ],
        "best_config_selected_holdouts": best_config["selected"] if best_config else [],
        "best_config_is_safe": best_config_is_safe,
        "recommended_validation": strip_vectors(recommended),
        "top_candidates": strip_vectors(candidates[: args.top_k]),
        "decision": "validate_recommended_candidates" if recommended else "no_classifier_safe_candidates",
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "label_rows": output["label_rows"],
        "holdout_scored_rows": output["holdout_scored_rows"],
        "true_strong_rows": output["true_strong_rows"],
        "configs_tested_with_coverage": output["configs_tested_with_coverage"],
        "best_config_is_safe": output["best_config_is_safe"],
        "recommended_validation": len(output["recommended_validation"]),
        "decision": output["decision"],
    }, indent=2))
    for row in output["top_candidates"][:5]:
        print(
            f"{row['match']} minute={row['cut_minute']} score={row['robust_boundary_score']:.3f} "
            f"p={row['predicted_robust_probability']:.3f} lcb={row['predicted_lcb90_pp']:.3f} "
            f"pos={row['predicted_positive_seed_rate']:.3f} downside={row['predicted_downside_pp']:.3f}"
        )


if __name__ == "__main__":
    main()
