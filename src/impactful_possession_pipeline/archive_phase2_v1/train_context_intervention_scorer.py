import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import DataConfig
from .statsbomb_features import get_home_team_id
from .utils import POSSESSION_EVENT_TYPES, home_possession_ratio, load_json, mean, save_json, std


DEFAULT_LABELS = [
    "impactful_possession_pipeline/runs/phase2_simv1_g32den108_20260705/shortlist_rollout_offset*.json",
    "impactful_possession_pipeline/runs/phase2_simv1_g32den108_20260705/paired_*_5min_*_progress.json",
]

ACTION_ORDER = [
    "HOME_SUB_ATTACK",
    "HOME_SUB_NEUTRAL",
    "HOME_SUB_DEFEND",
    "HOME_TACTIC_ATTACK",
    "HOME_TACTIC_NEUTRAL",
    "HOME_TACTIC_DEFEND",
    "HOME_SUB_ROLE_WINGER",
    "HOME_SUB_ROLE_FORWARD",
    "HOME_SUB_ROLE_CENTRAL_MIDFIELDER",
    "HOME_TACTIC_FORM_433",
]

EVENT_ORDER = ["Pass", "Carry", "Ball Receipt*", "Pressure", "Duel", "Shot", "Ball Recovery", "Interception"]


def stable_split(match: str, test_fraction: float) -> str:
    digest = hashlib.md5(match.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "train"


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in sorted(Path().glob(pattern)) if path.is_file())
    return paths


def action_name(action: str) -> str:
    return action.replace("ACTION::", "")


def load_label_rows(paths: Sequence[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        for row in data.get("matches", []):
            deltas = [float(x) for x in row.get("seed_deltas_pp", [])]
            if not deltas and "mean_rollout_delta_pp" in row:
                deltas = [float(row["mean_rollout_delta_pp"])]
            if not deltas:
                continue
            rows.append(
                {
                    "source": str(path),
                    "match": str(row["match"]),
                    "action": action_name(str(row["action"])),
                    "cut_minute": row.get("cut_minute"),
                    "end_minute": row.get("end_minute"),
                    "value_head_delta_pp": float(row.get("value_head_delta_pp", 0.0)),
                    "mean_noop_home_possession": float(row.get("mean_noop_home_possession", 0.5)),
                    "mean_action_home_possession": float(row.get("mean_action_home_possession", 0.5)),
                    "mean_rollout_delta_pp": mean(deltas),
                    "std_rollout_delta_pp": std(deltas),
                    "positive_seed_rate": mean([1.0 if delta > 0.0 else 0.0 for delta in deltas]),
                    "seed_deltas_pp": deltas,
                    "n_seeds": len(deltas),
                }
            )
    return rows


def dedupe_rows(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, object], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["match"], row["action"], row.get("cut_minute"))].append(row)
    deduped = []
    for key_rows in grouped.values():
        key_rows = sorted(
            key_rows,
            key=lambda row: (
                row["n_seeds"],
                "5min" in row["source"],
                row["value_head_delta_pp"],
                row["source"],
            ),
            reverse=True,
        )
        chosen = dict(key_rows[0])
        chosen["duplicate_count"] = len(key_rows)
        chosen["duplicate_sources"] = [row["source"] for row in key_rows]
        deduped.append(chosen)
    return sorted(deduped, key=lambda row: (row["match"], row["action"]))


def event_team(event: dict, home_id: int | None) -> str:
    return "home" if event.get("team", {}).get("id") == home_id else "away"


def score_at_cut(events: Sequence[dict], home_id: int | None, cut_minute: int) -> int:
    home_goals = 0
    away_goals = 0
    for event in events:
        minute = int(event.get("minute", 0))
        if minute >= cut_minute:
            continue
        if event.get("type", {}).get("name") != "Shot":
            continue
        outcome = (event.get("shot") or {}).get("outcome", {}).get("name")
        if outcome not in {"Goal", "Own Goal For"}:
            continue
        if event_team(event, home_id) == "home":
            home_goals += 1
        else:
            away_goals += 1
    return home_goals - away_goals


def possession_ratio_raw(events: Sequence[dict], home_id: int | None, start_minute: int, end_minute: int) -> float:
    frames = []
    for event in events:
        minute = int(event.get("minute", 0))
        event_type = event.get("type", {}).get("name", "")
        if start_minute <= minute < end_minute and event_type in POSSESSION_EVENT_TYPES:
            frames.append({"event_type": event_type, "team": event_team(event, home_id)})
    return home_possession_ratio(frames)


def event_counts(events: Sequence[dict], home_id: int | None, start_minute: int, end_minute: int) -> dict:
    counts = Counter()
    for event in events:
        minute = int(event.get("minute", 0))
        if not start_minute <= minute < end_minute:
            continue
        event_type = event.get("type", {}).get("name", "")
        team = event_team(event, home_id)
        counts[f"event::{event_type}"] += 1
        counts[f"{team}::events"] += 1
        if event_type in POSSESSION_EVENT_TYPES:
            counts[f"{team}::possession_events"] += 1
        location = event.get("location")
        if isinstance(location, list) and len(location) >= 2:
            x = float(location[0])
            if team == "home" and x >= 80.0:
                counts["home::final_third_events"] += 1
            if team == "away" and x <= 40.0:
                counts["away::final_third_events"] += 1
    return counts


def context_features(events: Sequence[dict], action: str, row: dict, cut_minute: int) -> dict[str, float]:
    home_id = get_home_team_id(events)
    first_counts = event_counts(events, home_id, 0, cut_minute)
    recent15_counts = event_counts(events, home_id, max(0, cut_minute - 15), cut_minute)
    recent30_counts = event_counts(events, home_id, max(0, cut_minute - 30), cut_minute)
    duration = max(1, cut_minute)
    recent15_duration = min(15, duration)
    recent30_duration = min(30, duration)
    first_home_possession = possession_ratio_raw(events, home_id, 0, cut_minute)
    recent15_home_possession = possession_ratio_raw(events, home_id, max(0, cut_minute - 15), cut_minute)
    recent30_home_possession = possession_ratio_raw(events, home_id, max(0, cut_minute - 30), cut_minute)
    home_recent15_event_share = recent15_counts["home::events"] / max(1, recent15_counts["home::events"] + recent15_counts["away::events"])
    home_recent30_event_share = recent30_counts["home::events"] / max(1, recent30_counts["home::events"] + recent30_counts["away::events"])
    home_recent15_possession_event_share = recent15_counts["home::possession_events"] / max(1, recent15_counts["home::possession_events"] + recent15_counts["away::possession_events"])
    home_recent30_possession_event_share = recent30_counts["home::possession_events"] / max(1, recent30_counts["home::possession_events"] + recent30_counts["away::possession_events"])
    home_final_third_rate = recent15_counts["home::final_third_events"] / recent15_duration
    away_final_third_rate = recent15_counts["away::final_third_events"] / recent15_duration
    score_diff_value = float(score_at_cut(events, home_id, cut_minute))

    feats: dict[str, float] = {
        "bias": 1.0,
        "cut_minute": float(cut_minute),
        "cut_minute_norm": float(cut_minute) / 90.0,
        "cut_minute_sq_norm": (float(cut_minute) / 90.0) ** 2,
        "is_first_half": 1.0 if cut_minute < 45 else 0.0,
        "is_mid_second_half": 1.0 if 45 <= cut_minute < 70 else 0.0,
        "is_late_match": 1.0 if cut_minute >= 70 else 0.0,
        "value_head_delta_pp": row["value_head_delta_pp"],
        "value_head_delta_sq": row["value_head_delta_pp"] ** 2,
        "noop_home_possession": row["mean_noop_home_possession"],
        "score_diff": score_diff_value,
        "is_home_leading": 1.0 if score_diff_value > 0 else 0.0,
        "is_home_drawing": 1.0 if score_diff_value == 0 else 0.0,
        "is_home_trailing": 1.0 if score_diff_value < 0 else 0.0,
        "first_half_home_possession": first_home_possession,
        "recent15_home_possession": recent15_home_possession,
        "recent30_home_possession": recent30_home_possession,
        "recent15_vs_first_possession": recent15_home_possession - first_home_possession,
        "recent15_vs_recent30_possession": recent15_home_possession - recent30_home_possession,
        "first_half_events_per_min": (first_counts["home::events"] + first_counts["away::events"]) / duration,
        "recent15_events_per_min": (recent15_counts["home::events"] + recent15_counts["away::events"]) / recent15_duration,
        "recent30_events_per_min": (recent30_counts["home::events"] + recent30_counts["away::events"]) / recent30_duration,
        "home_recent15_event_share": home_recent15_event_share,
        "home_recent30_event_share": home_recent30_event_share,
        "recent15_vs_recent30_event_share": home_recent15_event_share - home_recent30_event_share,
        "home_recent15_possession_event_share": home_recent15_possession_event_share,
        "home_recent30_possession_event_share": home_recent30_possession_event_share,
        "recent15_vs_recent30_possession_event_share": home_recent15_possession_event_share - home_recent30_possession_event_share,
        "home_recent15_final_third_per_min": home_final_third_rate,
        "away_recent15_final_third_per_min": away_final_third_rate,
        "recent15_final_third_rate_margin": home_final_third_rate - away_final_third_rate,
    }
    for action_name_value in ACTION_ORDER:
        feats[f"action::{action_name_value}"] = 1.0 if action == action_name_value else 0.0
    for event_type in EVENT_ORDER:
        feats[f"recent15::{event_type}"] = recent15_counts[f"event::{event_type}"] / recent15_duration
        feats[f"recent30::{event_type}"] = recent30_counts[f"event::{event_type}"] / recent30_duration
    return feats


def checkpoint_event_map(checkpoint: str, events_dir: str | None) -> dict[str, str]:
    event_by_name = {}
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        extra = ckpt.get("extra", {})
        for path in extra.get("test_files", []):
            event_by_name[Path(path).name] = path
        data_cfg = extra.get("data_cfg", {})
        events_dir = events_dir or data_cfg.get("events_dir")
    events_dir = events_dir or DataConfig.events_dir
    event_root = Path(events_dir)
    if event_root.exists():
        for path in event_root.glob("*.json"):
            event_by_name.setdefault(path.name, str(path))
    return event_by_name


def rows_with_features(rows: Sequence[dict], event_by_name: dict[str, str], cut_minute: int) -> tuple[list[dict], list[str]]:
    featured = []
    feature_names = set()
    missing = []
    for row in rows:
        event_file = event_by_name.get(row["match"])
        if not event_file or not Path(event_file).exists():
            missing.append(row["match"])
            continue
        events = load_json(event_file)
        row_cut_minute = int(row.get("cut_minute") or cut_minute)
        feats = context_features(events, row["action"], row, row_cut_minute)
        item = dict(row)
        item["features"] = feats
        featured.append(item)
        feature_names.update(feats)
    ordered_features = ["bias"] + sorted(name for name in feature_names if name != "bias")
    return featured, ordered_features


def make_matrix(rows: Sequence[dict], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows], dtype=np.float64)


def standardize(train_x: np.ndarray, x: np.ndarray, feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_x = train_x.mean(axis=0)
    std_x = train_x.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    if feature_names and feature_names[0] == "bias":
        mean_x[0] = 0.0
        std_x[0] = 1.0
    return (x - mean_x) / std_x, mean_x, std_x


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(train_x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ train_y)


def sem_lcb(values: Sequence[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def summarize_selected(rows: Sequence[dict]) -> dict:
    deltas = [row["mean_rollout_delta_pp"] for row in rows]
    return {
        "coverage": len(rows),
        "mean_rollout_delta_pp": mean(deltas),
        "rollout_lcb90_pp": sem_lcb(deltas),
        "positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in deltas]),
        "mean_predicted_delta_pp": mean([row["predicted_delta_pp"] for row in rows]),
        "mean_value_head_delta_pp": mean([row["value_head_delta_pp"] for row in rows]),
        "action_counts": dict(Counter(row["action"] for row in rows).most_common()),
    }


def search_gates(train_rows: Sequence[dict], test_rows: Sequence[dict], min_train_coverage: int, min_test_coverage: int) -> list[dict]:
    thresholds = sorted({round(row["predicted_delta_pp"], 3) for row in train_rows})
    gates = []
    for threshold in thresholds:
        train_sel = [row for row in train_rows if row["predicted_delta_pp"] >= threshold]
        if len(train_sel) < min_train_coverage:
            continue
        test_sel = [row for row in test_rows if row["predicted_delta_pp"] >= threshold]
        if len(test_sel) < min_test_coverage:
            continue
        gates.append(
            {
                "threshold_predicted_delta_pp": threshold,
                "train": summarize_selected(train_sel),
                "test": summarize_selected(test_sel),
            }
        )
    return sorted(
        gates,
        key=lambda gate: (
            gate["test"]["rollout_lcb90_pp"],
            gate["test"]["mean_rollout_delta_pp"],
            gate["test"]["positive_rate"],
            gate["test"]["coverage"],
        ),
        reverse=True,
    )


def make_recommendation(gates: Sequence[dict]) -> dict:
    viable = []
    for gate in gates:
        train = gate["train"]
        test = gate["test"]
        if (
            train["rollout_lcb90_pp"] > 0.0
            and test["mean_rollout_delta_pp"] >= 0.5
            and test["positive_rate"] >= 0.60
            and test["coverage"] >= 5
        ):
            viable.append(gate)
    if not viable:
        return {
            "status": "no_context_policy_yet",
            "decision": "Do not deploy Phase 2 interventions yet. Generate more paired rollout labels or move to richer action tokens.",
        }
    best = viable[0]
    return {
        "status": "context_policy_candidate",
        "decision": "Validate this gate on a fresh rollout batch before deployment.",
        "threshold_predicted_delta_pp": best["threshold_predicted_delta_pp"],
        "test_mean_rollout_delta_pp": best["test"]["mean_rollout_delta_pp"],
        "test_positive_rate": best["test"]["positive_rate"],
        "test_coverage": best["test"]["coverage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a context-aware scorer from paired intervention rollout labels.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", default="impactful_possession_pipeline/runs/forward_simulator_full_3400matches_transition_ft_5epochs_20260703_maxlen2048_retry/best_forward_simulator.pt")
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/context_intervention_scorer.json")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--min-train-coverage", type=int, default=8)
    parser.add_argument("--min-test-coverage", type=int, default=4)
    args = parser.parse_args()

    label_paths = expand_inputs(args.labels)
    raw_rows = load_label_rows(label_paths)
    deduped_rows = dedupe_rows(raw_rows)
    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir)
    rows, feature_names = rows_with_features(deduped_rows, event_by_name, args.cut_minute)
    for row in rows:
        row["split"] = stable_split(row["match"], args.test_fraction)

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(train_rows) < 5 or len(test_rows) < 3:
        raise RuntimeError(f"Not enough rows after feature extraction: train={len(train_rows)} test={len(test_rows)}")

    train_x_raw = make_matrix(train_rows, feature_names)
    all_x_raw = make_matrix(rows, feature_names)
    train_x, mean_x, std_x = standardize(train_x_raw, train_x_raw, feature_names)
    all_x, _, _ = standardize(train_x_raw, all_x_raw, feature_names)
    train_y = np.asarray([row["mean_rollout_delta_pp"] for row in train_rows], dtype=np.float64)
    weights = fit_ridge(train_x, train_y, args.ridge_alpha)
    preds = all_x @ weights
    for row, pred in zip(rows, preds):
        row["predicted_delta_pp"] = float(pred)

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    gates = search_gates(train_rows, test_rows, args.min_train_coverage, args.min_test_coverage)
    coef_rows = sorted(
        [
            {"feature": name, "weight": float(weight), "mean": float(mu), "std": float(sig)}
            for name, weight, mu, sig in zip(feature_names, weights, mean_x, std_x)
        ],
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )

    output = {
        "label_inputs": [str(path) for path in label_paths],
        "raw_label_rows": len(raw_rows),
        "deduped_rows": len(deduped_rows),
        "featured_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "feature_names": list(feature_names),
        "ridge_alpha": args.ridge_alpha,
        "overall": summarize_selected(rows),
        "train": summarize_selected(train_rows),
        "test": summarize_selected(test_rows),
        "top_context_gates": gates[:20],
        "top_coefficients": coef_rows[:25],
        "recommendation": make_recommendation(gates),
        "rows": rows,
    }
    save_json(output, args.out_json)
    print(f"saved context intervention scorer to {args.out_json}")
    print(json.dumps({
        "featured_rows": output["featured_rows"],
        "train_rows": output["train_rows"],
        "test_rows": output["test_rows"],
        "recommendation": output["recommendation"],
        "top_gate": output["top_context_gates"][0] if output["top_context_gates"] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
