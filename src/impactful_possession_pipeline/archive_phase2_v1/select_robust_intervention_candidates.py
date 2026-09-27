import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import DataConfig
from .train_context_intervention_scorer import (
    action_name,
    checkpoint_event_map,
    context_features,
    dedupe_rows,
    expand_inputs,
    load_label_rows,
)
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


def row_lcb(row: dict, z: float) -> float:
    return sem_lcb([float(x) for x in row["seed_deltas_pp"]], z)



def normalize_match_name(match: str) -> str:
    name = Path(str(match)).name
    return name if name.endswith(".json") else f"{name}.json"

def make_feature_names(rows: Sequence[dict]) -> list[str]:
    names = set()
    for row in rows:
        names.update(row["features"])
    excluded = {"value_head_delta_pp", "value_head_delta_sq", "noop_home_possession"}
    return ["bias"] + sorted(
        name for name in names if name != "bias" and not name.startswith("action::") and name not in excluded
    )


def make_matrix(rows: Sequence[dict], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    mu[0] = 0.0
    sigma[0] = 1.0
    return (x - mu) / sigma, mu, sigma


def apply_standardize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def min_distance(vector: np.ndarray, examples: Sequence[np.ndarray]) -> float:
    if not examples:
        return 0.0
    return min(float(np.linalg.norm(vector - example)) for example in examples)


def checkpoint_test_files(checkpoint: str) -> list[str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    return list(ckpt.get("extra", {}).get("test_files", []))


def make_eval_command(args: argparse.Namespace, candidates: Sequence[dict]) -> str:
    paths = " ".join(candidate["match_json"] for candidate in candidates)
    return (
        f"CUDA_VISIBLE_DEVICES={args.cuda_device} "
        f"/home/peter/miniconda3/envs/fb-env/bin/python -u -m "
        f"impactful_possession_pipeline.evaluate_forced_sequential_batch "
        f"--checkpoint {args.checkpoint} "
        f"--out-json {args.eval_out_json} "
        f"--match-jsons {paths} "
        f"--max-matches {len(candidates)} "
        f"--rollout-seeds {args.eval_rollout_seeds} "
        f"--action ACTION::{args.action} "
        f"--cut-minute {args.cut_minute} "
        f"--end-minute {args.end_minute} "
        f"--step-minutes {args.step_minutes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank robust Phase 2 intervention candidates by LCB/seed reliability.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="HOME_SUB_ROLE_CENTRAL_MIDFIELDER")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=50)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--z", type=float, default=1.64)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-validation-seeds", type=int, default=8)
    parser.add_argument("--robust-lcb-threshold", type=float, default=0.0)
    parser.add_argument("--robust-positive-seed-rate", type=float, default=0.625)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-eval-robust-score", type=float, default=0.0)
    parser.add_argument("--min-eval-predicted-lcb", type=float, default=0.0)
    parser.add_argument("--eval-top-k", type=int, default=2)
    parser.add_argument("--eval-rollout-seeds", type=int, default=4)
    parser.add_argument("--explore-top-k", type=int, default=2)
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--eval-out-json", default="impactful_possession_pipeline/runs/robust_candidate_eval.json")
    args = parser.parse_args()

    label_paths = expand_inputs(args.labels)
    rows = [
        row
        for row in dedupe_rows(load_label_rows(label_paths))
        if action_name(row["action"]) == args.action
    ]
    if not rows:
        raise RuntimeError(f"no label rows found for action={args.action}")

    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    featured = []
    for row in rows:
        event_file = event_by_name.get(row["match"])
        if not event_file or not Path(event_file).exists():
            continue
        item = dict(row)
        item["lcb90_pp"] = row_lcb(item, args.z)
        item_cut_minute = int(item.get("cut_minute") or args.cut_minute)
        item["features"] = context_features(load_json(event_file), item["action"], item, item_cut_minute)
        featured.append(item)
    if len(featured) < 5:
        raise RuntimeError(f"not enough featured rows: {len(featured)}")

    validated = [row for row in featured if row["n_seeds"] >= args.min_validation_seeds]
    robust_rows = [
        row
        for row in validated
        if row["lcb90_pp"] > args.robust_lcb_threshold
        and row["positive_seed_rate"] >= args.robust_positive_seed_rate
    ]
    failed_rows = [
        row
        for row in featured
        if row["lcb90_pp"] <= args.robust_lcb_threshold
        or row["positive_seed_rate"] < args.robust_positive_seed_rate
    ]
    if not robust_rows:
        raise RuntimeError("no robust validation anchors found; lower thresholds or generate more 8-seed labels")

    feature_names = make_feature_names(featured)
    x_raw = make_matrix(featured, feature_names)
    x, mu, sigma = standardize(x_raw)
    y_lcb = np.asarray([row["lcb90_pp"] for row in featured], dtype=np.float64)
    y_rate = np.asarray([row["positive_seed_rate"] for row in featured], dtype=np.float64)
    y_std = np.asarray([row["std_rollout_delta_pp"] for row in featured], dtype=np.float64)
    lcb_weights = fit_ridge(x, y_lcb, args.ridge_alpha)
    rate_weights = fit_ridge(x, y_rate, args.ridge_alpha)
    std_weights = fit_ridge(x, y_std, args.ridge_alpha)

    row_by_key = {(normalize_match_name(row["match"]), action_name(row["action"]), row.get("cut_minute")): row for row in featured}
    robust_vectors = [
        apply_standardize(make_matrix([row], feature_names), mu, sigma)[0]
        for row in robust_rows
    ]
    failed_vectors = [
        apply_standardize(make_matrix([row], feature_names), mu, sigma)[0]
        for row in failed_rows
    ]

    candidate_rows = []
    base_row = {
        "value_head_delta_pp": 0.0,
        "mean_noop_home_possession": 0.5,
        "action": args.action,
    }
    for match_json in checkpoint_test_files(args.checkpoint):
        match = Path(match_json).name
        key = (match, args.action, args.cut_minute)
        if key in row_by_key:
            continue
        events = load_json(match_json)
        feats = context_features(events, args.action, base_row, args.cut_minute)
        matrix = make_matrix([{"features": feats}], feature_names)
        vector = apply_standardize(matrix, mu, sigma)[0]
        robust_dist = min_distance(vector, robust_vectors)
        failed_dist = min_distance(vector, failed_vectors)
        pred_lcb = float(vector @ lcb_weights)
        pred_rate = float(vector @ rate_weights)
        pred_std = max(0.0, float(vector @ std_weights))
        distance_margin = failed_dist - robust_dist
        robust_score = pred_lcb + 8.0 * (pred_rate - 0.5) + 0.75 * distance_margin - 0.10 * robust_dist
        candidate_rows.append(
            {
                "match": match,
                "match_json": match_json,
                "action": args.action,
                "robust_score": robust_score,
                "predicted_lcb90_pp": pred_lcb,
                "predicted_positive_seed_rate": pred_rate,
                "predicted_std_delta_pp": pred_std,
                "robust_anchor_distance": robust_dist,
                "failed_anchor_distance": failed_dist,
                "distance_margin": distance_margin,
                "features": feats,
            }
        )

    candidate_rows = sorted(candidate_rows, key=lambda row: row["robust_score"], reverse=True)
    eval_eligible = [
        row
        for row in candidate_rows
        if row["robust_score"] >= args.min_eval_robust_score
        and row["predicted_lcb90_pp"] >= args.min_eval_predicted_lcb
    ]
    top_eval = eval_eligible[: args.eval_top_k]
    explore_eval = candidate_rows[: args.explore_top_k]
    output = {
        "label_inputs": [str(path) for path in label_paths],
        "action": args.action,
        "cut_minute": args.cut_minute,
        "featured_rows": len(featured),
        "validated_rows": len(validated),
        "robust_rows": [
            {
                "match": row["match"],
                "mean_rollout_delta_pp": row["mean_rollout_delta_pp"],
                "std_rollout_delta_pp": row["std_rollout_delta_pp"],
                "positive_seed_rate": row["positive_seed_rate"],
                "lcb90_pp": row["lcb90_pp"],
            }
            for row in robust_rows
        ],
        "failed_or_noisy_rows": [
            {
                "match": row["match"],
                "mean_rollout_delta_pp": row["mean_rollout_delta_pp"],
                "std_rollout_delta_pp": row["std_rollout_delta_pp"],
                "positive_seed_rate": row["positive_seed_rate"],
                "lcb90_pp": row["lcb90_pp"],
            }
            for row in failed_rows
        ],
        "feature_names": feature_names,
        "top_candidates": candidate_rows[: args.max_candidates],
        "recommended_eval": {
            "top_k": len(top_eval),
            "requested_top_k": args.eval_top_k,
            "rollout_seeds": args.eval_rollout_seeds,
            "min_eval_robust_score": args.min_eval_robust_score,
            "min_eval_predicted_lcb": args.min_eval_predicted_lcb,
            "status": "ready" if top_eval else "no_positive_lcb_candidates",
            "command": make_eval_command(args, top_eval) if top_eval else None,
        },
        "recommended_exploration_eval": {
            "top_k": len(explore_eval),
            "requested_top_k": args.explore_top_k,
            "rollout_seeds": args.eval_rollout_seeds,
            "status": "ready" if explore_eval else "no_candidates",
            "note": "Exploration only; not a deployable recommendation unless rollout LCB confirms positive.",
            "command": make_eval_command(args, explore_eval) if explore_eval else None,
        },
    }
    save_json(output, args.out_json)
    print(f"saved robust candidate selector to {args.out_json}")
    print(
        json.dumps(
            {
                "featured_rows": output["featured_rows"],
                "validated_rows": output["validated_rows"],
                "robust_rows": output["robust_rows"],
                "top_candidates": output["top_candidates"][: min(5, len(output["top_candidates"]))],
                "recommended_eval": output["recommended_eval"],
                "recommended_exploration_eval": output["recommended_exploration_eval"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
