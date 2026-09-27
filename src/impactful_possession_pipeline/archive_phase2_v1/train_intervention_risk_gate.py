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


def checkpoint_test_files(checkpoint: str) -> list[str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    return list(ckpt.get("extra", {}).get("test_files", []))


def parse_minutes(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


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
        item = dict(row)
        item["cut_minute"] = int(item.get("cut_minute") or default_cut_minute)
        item["lcb90_pp"] = sem_lcb([float(x) for x in item["seed_deltas_pp"]])
        item["robust_label"] = int(item["lcb90_pp"] > 0.0 and item["positive_seed_rate"] >= 0.625)
        item["features"] = context_features(load_json(event_file), item["action"], item, item["cut_minute"])
        featured.append(item)
    return featured



def min_distance(vector: np.ndarray, examples: Sequence[np.ndarray]) -> float:
    if not examples:
        return 0.0
    return min(float(np.linalg.norm(vector - example)) for example in examples)

def predict_rows(rows: list[dict], names: Sequence[str], mu: np.ndarray, sigma: np.ndarray, weights: dict[str, np.ndarray], robust_vectors: Sequence[np.ndarray] | None = None, failed_vectors: Sequence[np.ndarray] | None = None) -> None:
    x = apply_standardize(make_matrix(rows, names), mu, sigma)
    robust_vectors = list(robust_vectors or [])
    failed_vectors = list(failed_vectors or [])
    for index, row in enumerate(rows):
        vector = x[index]
        robust_distance = min_distance(vector, robust_vectors)
        failed_distance = min_distance(vector, failed_vectors)
        distance_margin = failed_distance - robust_distance
        row["robust_anchor_distance"] = robust_distance
        row["failed_anchor_distance"] = failed_distance
        row["anchor_distance_margin"] = distance_margin
        row["predicted_lcb90_pp"] = float(vector @ weights["lcb"])
        row["predicted_positive_seed_rate"] = float(vector @ weights["positive_rate"])
        row["predicted_std_delta_pp"] = max(0.0, float(vector @ weights["std"]))
        row["risk_gate_score"] = (
            row["predicted_lcb90_pp"]
            + 8.0 * (row["predicted_positive_seed_rate"] - 0.625)
            - 0.10 * row["predicted_std_delta_pp"]
            + 2.0 * distance_margin
            - 0.50 * robust_distance
        )


def make_candidate_rows(
    checkpoint: str,
    event_by_name: dict[str, str],
    action: str,
    minutes: Sequence[int],
    max_matches: int,
) -> list[dict]:
    rows = []
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
            feats = context_features(events, action, base_row, minute)
            rows.append(
                {
                    "match": Path(event_file).name,
                    "match_json": event_file,
                    "action": f"ACTION::{action}",
                    "cut_minute": minute,
                    "end_minute": minute + 5,
                    "features": feats,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conservative intervention risk gate from rollout labels.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="HOME_SUB_ROLE_CENTRAL_MIDFIELDER")
    parser.add_argument("--default-cut-minute", type=int, default=45)
    parser.add_argument("--candidate-minutes", default="40,45,50,55")
    parser.add_argument("--max-candidate-matches", type=int, default=500)
    parser.add_argument("--ridge-alpha", type=float, default=25.0)
    parser.add_argument("--top-k", type=int, default=20)
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
        raise RuntimeError(f"not enough labels for risk gate: {len(label_rows)}")

    names = feature_names(label_rows)
    x_raw = make_matrix(label_rows, names)
    x, mu, sigma = standardize(x_raw)
    weights = {
        "lcb": fit_ridge(x, np.asarray([row["lcb90_pp"] for row in label_rows]), args.ridge_alpha),
        "positive_rate": fit_ridge(x, np.asarray([row["positive_seed_rate"] for row in label_rows]), args.ridge_alpha),
        "std": fit_ridge(x, np.asarray([row["std_rollout_delta_pp"] for row in label_rows]), args.ridge_alpha),
    }
    robust_label_rows = [row for row in label_rows if row["robust_label"]]
    failed_label_rows = [row for row in label_rows if not row["robust_label"]]
    robust_vectors = [x[index] for index, row in enumerate(label_rows) if row["robust_label"]]
    failed_vectors = [x[index] for index, row in enumerate(label_rows) if not row["robust_label"]]
    predict_rows(label_rows, names, mu, sigma, weights, robust_vectors, failed_vectors)

    minutes = parse_minutes(args.candidate_minutes)
    candidate_rows = make_candidate_rows(
        args.checkpoint,
        event_by_name,
        args.action,
        minutes,
        args.max_candidate_matches,
    )
    seen = {(row["match"], f"ACTION::{row['action']}", row.get("cut_minute")) for row in label_rows}
    candidate_rows = [row for row in candidate_rows if (row["match"], row["action"], row["cut_minute"]) not in seen]
    predict_rows(candidate_rows, names, mu, sigma, weights, robust_vectors, failed_vectors)
    candidate_rows = sorted(candidate_rows, key=lambda row: (row["risk_gate_score"], row["predicted_lcb90_pp"]), reverse=True)
    eval_ready = [
        row
        for row in candidate_rows
        if row["predicted_lcb90_pp"] > 0.0
        and row["predicted_positive_seed_rate"] >= 0.625
        and row["risk_gate_score"] > 0.0
        and row["anchor_distance_margin"] > 0.0
    ]

    robust = robust_label_rows
    failed = failed_label_rows
    output = {
        "label_inputs": [str(path) for path in label_paths],
        "action": args.action,
        "label_rows": len(label_rows),
        "robust_label_rows": len(robust),
        "failed_or_noisy_label_rows": len(failed),
        "feature_names": names,
        "training_summary": {
            "mean_lcb90_pp": mean([row["lcb90_pp"] for row in label_rows]),
            "robust_rate": mean([row["robust_label"] for row in label_rows]),
            "mean_positive_seed_rate": mean([row["positive_seed_rate"] for row in label_rows]),
        },
        "top_label_predictions": sorted(
            [
                {
                    "match": row["match"],
                    "cut_minute": row.get("cut_minute"),
                    "lcb90_pp": row["lcb90_pp"],
                    "positive_seed_rate": row["positive_seed_rate"],
                    "predicted_lcb90_pp": row["predicted_lcb90_pp"],
                    "risk_gate_score": row["risk_gate_score"],
                    "robust_anchor_distance": row["robust_anchor_distance"],
                    "failed_anchor_distance": row["failed_anchor_distance"],
                    "anchor_distance_margin": row["anchor_distance_margin"],
                    "robust_label": row["robust_label"],
                }
                for row in label_rows
            ],
            key=lambda row: row["risk_gate_score"],
            reverse=True,
        ),
        "top_candidates": candidate_rows[: args.top_k],
        "eval_ready_candidates": eval_ready[: args.top_k],
        "recommendation": {
            "status": "ready" if eval_ready else "no_positive_lcb_candidates",
            "top_k": min(args.top_k, len(eval_ready)),
        },
    }
    save_json(output, args.out_json)
    print(json.dumps(output["training_summary"], indent=2))
    print(json.dumps(output["recommendation"], indent=2))
    for row in output["top_candidates"][:5]:
        print(
            f"{row['match']} minute={row['cut_minute']} "
            f"score={row['risk_gate_score']:.3f} lcb={row['predicted_lcb90_pp']:.3f} "
            f"pos={row['predicted_positive_seed_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
