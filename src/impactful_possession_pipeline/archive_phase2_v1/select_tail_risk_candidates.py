import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import DataConfig
from .train_context_intervention_scorer import (
    ACTION_ORDER,
    checkpoint_event_map,
    context_features,
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


def cvar_bottom(values: Sequence[float], fraction: float = 0.25) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    count = max(1, math.ceil(len(ordered) * fraction))
    return mean(ordered[:count])


def normalize_match_name(match: str) -> str:
    name = Path(str(match)).name
    return name if name.endswith(".json") else f"{name}.json"


def combine_label_rows(rows: Sequence[dict], default_cut_minute: int, default_horizon: int) -> list[dict]:
    grouped: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        cut_minute = int(row.get("cut_minute") or default_cut_minute)
        end_minute = int(row.get("end_minute") or cut_minute + default_horizon)
        grouped[(normalize_match_name(row["match"]), row["action"], cut_minute, end_minute)].append(row)

    combined = []
    for (match, action, cut_minute, end_minute), key_rows in grouped.items():
        deltas = [float(delta) for row in key_rows for delta in row["seed_deltas_pp"]]
        noop_values = [row["mean_noop_home_possession"] for row in key_rows]
        action_values = [row["mean_action_home_possession"] for row in key_rows]
        combined.append(
            {
                "match": match,
                "action": action,
                "cut_minute": cut_minute,
                "end_minute": end_minute,
                "value_head_delta_pp": mean([row["value_head_delta_pp"] for row in key_rows]),
                "mean_noop_home_possession": mean(noop_values),
                "mean_action_home_possession": mean(action_values),
                "mean_rollout_delta_pp": mean(deltas),
                "std_rollout_delta_pp": std(deltas),
                "lcb90_pp": sem_lcb(deltas),
                "positive_seed_rate": mean([1.0 if delta > 0.0 else 0.0 for delta in deltas]),
                "worst_seed_delta_pp": min(deltas),
                "cvar25_delta_pp": cvar_bottom(deltas, 0.25),
                "seed_deltas_pp": deltas,
                "n_seeds": len(deltas),
                "sources": sorted({row["source"] for row in key_rows}),
            }
        )
    return sorted(combined, key=lambda row: (row["match"], row["action"], row["cut_minute"]))


def checkpoint_test_files(checkpoint: str) -> list[str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    return list(ckpt.get("extra", {}).get("test_files", []))


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_minutes(text: str) -> list[int]:
    return [int(part) for part in parse_csv(text)]


def make_feature_names(rows: Sequence[dict]) -> list[str]:
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


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def pearson_corr(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) < 2:
        return 0.0
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if float(a.std()) < 1e-8 or float(p.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, p)[0, 1])


def hashed_fold(key: str, folds: int) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, folds)


def summarize_regression(actual: Sequence[float], predicted: Sequence[float]) -> dict:
    errors = [float(pred) - float(act) for act, pred in zip(actual, predicted)]
    return {
        "n": len(errors),
        "mae": mean([abs(error) for error in errors]),
        "bias": mean(errors),
        "rmse": math.sqrt(mean([error * error for error in errors])) if errors else 0.0,
        "pearson": pearson_corr(actual, predicted),
    }


def tail_safe(row: dict, min_lcb: float, min_cvar25: float, min_worst: float, min_positive_rate: float) -> bool:
    return (
        float(row["lcb90_pp"]) >= min_lcb
        and float(row["cvar25_delta_pp"]) >= min_cvar25
        and float(row["worst_seed_delta_pp"]) >= min_worst
        and float(row["positive_seed_rate"]) >= min_positive_rate
    )


def predicted_tail_safe(row: dict, min_lcb: float, min_cvar25: float, min_worst: float, min_positive_rate: float) -> bool:
    return (
        float(row["predicted_lcb"]) >= min_lcb
        and float(row["predicted_cvar25"]) >= min_cvar25
        and float(row["predicted_worst"]) >= min_worst
        and float(row["predicted_positive_rate"]) >= min_positive_rate
    )


def cross_validate_tail_models(
    rows: Sequence[dict],
    names: Sequence[str],
    alpha: float,
    folds: int,
    min_lcb: float,
    min_cvar25: float,
    min_worst: float,
    min_positive_rate: float,
) -> dict:
    if folds < 2 or len(rows) < 10:
        return {"status": "skipped", "reason": "not enough rows or folds"}

    predictions: list[dict] = []
    target_getters = {
        "mean": "mean_rollout_delta_pp",
        "lcb": "lcb90_pp",
        "positive_rate": "positive_seed_rate",
        "std": "std_rollout_delta_pp",
        "worst": "worst_seed_delta_pp",
        "cvar25": "cvar25_delta_pp",
    }
    all_x = make_matrix(rows, names)
    for fold in range(folds):
        train_idx = [i for i, row in enumerate(rows) if hashed_fold(str(row["match"]), folds) != fold]
        val_idx = [i for i, row in enumerate(rows) if hashed_fold(str(row["match"]), folds) == fold]
        if len(train_idx) < 8 or not val_idx:
            continue
        train_raw = all_x[train_idx]
        val_raw = all_x[val_idx]
        train_x, mu, sigma = standardize(train_raw)
        val_x = apply_standardize(val_raw, mu, sigma)
        fold_weights = {}
        for target_name, row_key in target_getters.items():
            train_y = np.asarray([float(rows[i][row_key]) for i in train_idx], dtype=np.float64)
            fold_weights[target_name] = fit_ridge(train_x, train_y, alpha)
        for local_index, row_index in enumerate(val_idx):
            row = rows[row_index]
            vector = val_x[local_index]
            item = {
                "fold": fold,
                "match": row["match"],
                "action": f"ACTION::{row['action']}",
                "cut_minute": row["cut_minute"],
                "end_minute": row["end_minute"],
                "actual_tail_safe": tail_safe(row, min_lcb, min_cvar25, min_worst, min_positive_rate),
            }
            for target_name, row_key in target_getters.items():
                item[f"actual_{target_name}"] = float(row[row_key])
                item[f"predicted_{target_name}"] = float(vector @ fold_weights[target_name])
            item["predicted_tail_safe"] = predicted_tail_safe(item, min_lcb, min_cvar25, min_worst, min_positive_rate)
            predictions.append(item)

    if not predictions:
        return {"status": "skipped", "reason": "no non-empty validation folds"}

    target_metrics = {}
    for target_name in target_getters:
        target_metrics[target_name] = summarize_regression(
            [row[f"actual_{target_name}"] for row in predictions],
            [row[f"predicted_{target_name}"] for row in predictions],
        )
    tp = sum(1 for row in predictions if row["actual_tail_safe"] and row["predicted_tail_safe"])
    fp = sum(1 for row in predictions if not row["actual_tail_safe"] and row["predicted_tail_safe"])
    tn = sum(1 for row in predictions if not row["actual_tail_safe"] and not row["predicted_tail_safe"])
    fn = sum(1 for row in predictions if row["actual_tail_safe"] and not row["predicted_tail_safe"])
    return {
        "status": "complete",
        "folds": folds,
        "rows": len(predictions),
        "target_metrics": target_metrics,
        "tail_safe_gate": {
            "actual_safe_rows": tp + fn,
            "predicted_safe_rows": tp + fp,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
        },
        "worst_false_positive_rows": sorted(
            [row for row in predictions if row["predicted_tail_safe"] and not row["actual_tail_safe"]],
            key=lambda row: (row["actual_cvar25"], row["actual_worst"]),
        )[:10],
    }


def min_distance(vector: np.ndarray, examples: Sequence[np.ndarray]) -> float:
    if not examples:
        return 0.0
    return min(float(np.linalg.norm(vector - example)) for example in examples)


def add_features(rows: Sequence[dict], event_by_name: dict[str, str]) -> list[dict]:
    featured = []
    events_cache: dict[str, list[dict]] = {}
    for row in rows:
        event_file = event_by_name.get(row["match"])
        if not event_file or not Path(event_file).exists():
            continue
        if event_file not in events_cache:
            events_cache[event_file] = load_json(event_file)
        item = dict(row)
        item["features"] = context_features(events_cache[event_file], item["action"], item, int(item["cut_minute"]))
        featured.append(item)
    return featured


def make_candidate_rows(
    checkpoint: str,
    event_by_name: dict[str, str],
    actions: Sequence[str],
    minutes: Sequence[int],
    max_matches: int,
    horizon_minutes: int,
) -> list[dict]:
    candidates = []
    base_row = {
        "value_head_delta_pp": 0.0,
        "mean_noop_home_possession": 0.5,
    }
    for match_json in checkpoint_test_files(checkpoint)[:max_matches]:
        event_file = event_by_name.get(Path(match_json).name, match_json)
        if not Path(event_file).exists():
            continue
        events = load_json(event_file)
        for action in actions:
            row_base = {**base_row, "action": action}
            for minute in minutes:
                candidates.append(
                    {
                        "match": Path(event_file).name,
                        "match_json": event_file,
                        "action": f"ACTION::{action}",
                        "cut_minute": minute,
                        "end_minute": minute + horizon_minutes,
                        "step_minutes": horizon_minutes,
                        "features": context_features(events, action, row_base, minute),
                    }
                )
    return candidates


def make_eval_command(args: argparse.Namespace, candidates: Sequence[dict]) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={args.cuda_device} "
        f"/home/peter/miniconda3/envs/fb-env/bin/python -u -m "
        f"impactful_possession_pipeline.evaluate_forced_timing_candidates "
        f"--checkpoint {args.checkpoint} "
        f"--candidates-json {args.out_json} "
        f"--out-json {args.eval_out_json} "
        f"--top-k {len(candidates)} "
        f"--rollout-seeds {args.eval_rollout_seeds} "
        f"--seed {args.eval_seed} "
        f"--step-minutes {args.horizon_minutes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank intervention candidates by explicit downside-tail risk.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--actions", default=",".join(ACTION_ORDER))
    parser.add_argument("--candidate-minutes", default="40,45,50,55")
    parser.add_argument("--default-cut-minute", type=int, default=45)
    parser.add_argument("--horizon-minutes", type=int, default=5)
    parser.add_argument("--max-candidate-matches", type=int, default=500)
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    parser.add_argument("--min-predicted-lcb", type=float, default=0.0)
    parser.add_argument("--min-predicted-cvar25", type=float, default=-5.0)
    parser.add_argument("--min-predicted-worst", type=float, default=-25.0)
    parser.add_argument("--min-predicted-positive-rate", type=float, default=0.625)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--eval-top-k", type=int, default=2)
    parser.add_argument("--eval-rollout-seeds", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=91000)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--eval-out-json", default="impactful_possession_pipeline/runs/tail_risk_candidate_eval.json")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    label_rows = combine_label_rows(
        load_label_rows(expand_inputs(args.labels)),
        args.default_cut_minute,
        args.horizon_minutes,
    )
    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    featured = add_features(label_rows, event_by_name)
    if len(featured) < 8:
        raise RuntimeError(f"not enough featured labels: {len(featured)}")

    names = make_feature_names(featured)
    x_raw = make_matrix(featured, names)
    x, mu, sigma = standardize(x_raw)
    targets = {
        "mean": np.asarray([row["mean_rollout_delta_pp"] for row in featured], dtype=np.float64),
        "lcb": np.asarray([row["lcb90_pp"] for row in featured], dtype=np.float64),
        "positive_rate": np.asarray([row["positive_seed_rate"] for row in featured], dtype=np.float64),
        "std": np.asarray([row["std_rollout_delta_pp"] for row in featured], dtype=np.float64),
        "worst": np.asarray([row["worst_seed_delta_pp"] for row in featured], dtype=np.float64),
        "cvar25": np.asarray([row["cvar25_delta_pp"] for row in featured], dtype=np.float64),
    }
    weights = {name: fit_ridge(x, values, args.ridge_alpha) for name, values in targets.items()}
    cross_validation = cross_validate_tail_models(
        featured,
        names,
        args.ridge_alpha,
        args.cv_folds,
        args.min_predicted_lcb,
        args.min_predicted_cvar25,
        args.min_predicted_worst,
        args.min_predicted_positive_rate,
    )

    robust_vectors = [
        x[index]
        for index, row in enumerate(featured)
        if row["lcb90_pp"] > 0.0 and row["positive_seed_rate"] >= 0.625 and row["cvar25_delta_pp"] > -10.0
    ]
    failed_vectors = [
        x[index]
        for index, row in enumerate(featured)
        if row["lcb90_pp"] <= 0.0 or row["positive_seed_rate"] < 0.625 or row["cvar25_delta_pp"] <= -10.0
    ]

    actions = parse_csv(args.actions)
    minutes = parse_minutes(args.candidate_minutes)
    candidates = make_candidate_rows(
        args.checkpoint,
        event_by_name,
        actions,
        minutes,
        args.max_candidate_matches,
        args.horizon_minutes,
    )
    seen = {(row["match"], f"ACTION::{row['action']}", row["cut_minute"], row["end_minute"]) for row in featured}
    candidates = [
        row for row in candidates
        if (row["match"], row["action"], row["cut_minute"], row["end_minute"]) not in seen
    ]

    candidate_x = apply_standardize(make_matrix(candidates, names), mu, sigma)
    for index, row in enumerate(candidates):
        vector = candidate_x[index]
        robust_distance = min_distance(vector, robust_vectors)
        failed_distance = min_distance(vector, failed_vectors)
        row["robust_anchor_distance"] = robust_distance
        row["failed_anchor_distance"] = failed_distance
        row["distance_margin"] = failed_distance - robust_distance
        for target_name, target_weights in weights.items():
            row[f"predicted_{target_name}"] = float(vector @ target_weights)
        row["tail_risk_score"] = (
            row["predicted_lcb"]
            + 0.50 * row["predicted_cvar25"]
            + 0.15 * row["predicted_worst"]
            + 8.0 * (row["predicted_positive_rate"] - 0.625)
            - 0.05 * row["predicted_std"]
            + 0.75 * row["distance_margin"]
            - 0.10 * row["robust_anchor_distance"]
        )

    candidates = sorted(candidates, key=lambda row: (row["tail_risk_score"], row["predicted_lcb"]), reverse=True)
    eval_ready = [
        row for row in candidates
        if row["predicted_lcb"] >= args.min_predicted_lcb
        and row["predicted_cvar25"] >= args.min_predicted_cvar25
        and row["predicted_worst"] >= args.min_predicted_worst
        and row["predicted_positive_rate"] >= args.min_predicted_positive_rate
    ][: args.eval_top_k]

    output = {
        "label_inputs": [str(path) for path in expand_inputs(args.labels)],
        "checkpoint": args.checkpoint,
        "candidate_minutes": minutes,
        "actions": actions,
        "label_rows": len(featured),
        "robust_tail_rows": len(robust_vectors),
        "failed_tail_rows": len(failed_vectors),
        "cross_validation": cross_validation,
        "training_summary": {
            "mean_delta_pp": mean([row["mean_rollout_delta_pp"] for row in featured]),
            "mean_lcb90_pp": mean([row["lcb90_pp"] for row in featured]),
            "mean_cvar25_pp": mean([row["cvar25_delta_pp"] for row in featured]),
            "mean_worst_pp": mean([row["worst_seed_delta_pp"] for row in featured]),
            "mean_positive_seed_rate": mean([row["positive_seed_rate"] for row in featured]),
        },
        "top_labeled_rows": sorted(
            [
                {
                    "match": row["match"],
                    "action": f"ACTION::{row['action']}",
                    "cut_minute": row["cut_minute"],
                    "end_minute": row["end_minute"],
                    "n_seeds": row["n_seeds"],
                    "mean_rollout_delta_pp": row["mean_rollout_delta_pp"],
                    "lcb90_pp": row["lcb90_pp"],
                    "cvar25_delta_pp": row["cvar25_delta_pp"],
                    "worst_seed_delta_pp": row["worst_seed_delta_pp"],
                    "positive_seed_rate": row["positive_seed_rate"],
                }
                for row in featured
            ],
            key=lambda row: (row["lcb90_pp"], row["cvar25_delta_pp"]),
            reverse=True,
        )[: args.top_k],
        "top_candidates": candidates[: args.top_k],
        "eval_ready_candidates": eval_ready,
        "recommended_eval": {
            "status": "ready" if eval_ready else "no_tail_safe_candidates",
            "top_k": len(eval_ready),
            "rollout_seeds": args.eval_rollout_seeds,
            "command": make_eval_command(args, eval_ready) if eval_ready else None,
        },
    }
    save_json(output, args.out_json)
    print(json.dumps(output["training_summary"], indent=2))
    print(json.dumps(output["recommended_eval"], indent=2))
    for row in output["top_candidates"][:5]:
        print(
            f"{row['match']} {row['action']} {row['cut_minute']}-{row['end_minute']} "
            f"score={row['tail_risk_score']:.3f} lcb={row['predicted_lcb']:.3f} "
            f"cvar={row['predicted_cvar25']:.3f} worst={row['predicted_worst']:.3f} "
            f"pos={row['predicted_positive_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
