import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from .utils import mean, save_json, std


DEFAULT_INPUTS = [
    "impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/shortlisted_rollout_chunks/aggregate_000_020.json",
    "impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/shortlisted_rollout_strict_vh025/aggregate_strict_vh025.json",
    "impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/tactic_neutral_score_vh025/aggregate_tactic_neutral_vh025.json",
    "impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/tactic_neutral_score_vh025_seeds6/aggregate_tactic_neutral_vh025_seeds6.json",
    "impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/calibration_labels_vh025/*.json",
    "impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/calibration_labels_vh015_s1/*.json",
]


def action_name(action: str) -> str:
    return action.replace("ACTION::", "")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - pos) + ordered[high] * (pos - low))


def sem_lcb(values: list[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return mean(values) - z * std(values) / math.sqrt(len(values))


def stable_split(match: str, test_fraction: float) -> str:
    digest = hashlib.md5(match.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "train"


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(Path().glob(pattern))
        paths.extend(path for path in matched if path.is_file())
    return paths


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        if path.name == "action_calibration_summary.json":
            continue
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
                    "value_head_delta_pp": float(row.get("value_head_delta_pp", 0.0)),
                    "mean_rollout_delta_pp": mean(deltas),
                    "std_rollout_delta_pp": std(deltas),
                    "positive_seed_rate": mean([1.0 if x > 0.0 else 0.0 for x in deltas]),
                    "seed_deltas_pp": deltas,
                    "n_seeds": len(deltas),
                }
            )
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["match"], row["action"])].append(row)

    deduped = []
    for key_rows in grouped.values():
        # Prefer the most stable label when the same match/action was rerun.
        key_rows = sorted(
            key_rows,
            key=lambda row: (
                row["n_seeds"],
                "seeds6" in row["source"],
                "calibration_labels" in row["source"],
                row["source"],
            ),
            reverse=True,
        )
        chosen = dict(key_rows[0])
        chosen["duplicate_sources"] = [row["source"] for row in key_rows]
        chosen["duplicate_count"] = len(key_rows)
        deduped.append(chosen)
    return sorted(deduped, key=lambda row: (row["action"], row["match"]))


def summarize_rows(rows: list[dict]) -> dict:
    by_action: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_action[row["action"]].append(row)

    summary = {}
    for action, action_rows in sorted(by_action.items()):
        deltas = [row["mean_rollout_delta_pp"] for row in action_rows]
        vh = [row["value_head_delta_pp"] for row in action_rows]
        gaps = [row["mean_rollout_delta_pp"] - row["value_head_delta_pp"] for row in action_rows]
        summary[action] = {
            "n": len(action_rows),
            "mean_value_head_delta_pp": mean(vh),
            "mean_rollout_delta_pp": mean(deltas),
            "median_rollout_delta_pp": percentile(deltas, 0.5),
            "rollout_lcb90_pp": sem_lcb(deltas),
            "match_positive_rate": mean([1.0 if x > 0.0 else 0.0 for x in deltas]),
            "mean_positive_seed_rate": mean([row["positive_seed_rate"] for row in action_rows]),
            "mean_calibration_gap_pp": mean(gaps),
        }
    return summary


def candidate_thresholds(rows: list[dict]) -> list[float]:
    values = sorted({round(row["value_head_delta_pp"], 3) for row in rows})
    defaults = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    return sorted(set(defaults + values))


def evaluate_policy(rows: list[dict], action: str, threshold: float) -> dict:
    selected = [row for row in rows if row["action"] == action and row["value_head_delta_pp"] >= threshold]
    deltas = [row["mean_rollout_delta_pp"] for row in selected]
    return {
        "action": action,
        "threshold_value_head_delta_pp": threshold,
        "coverage": len(selected),
        "coverage_rate": len(selected) / len(rows) if rows else 0.0,
        "mean_rollout_delta_pp": mean(deltas),
        "rollout_lcb90_pp": sem_lcb(deltas),
        "positive_rate": mean([1.0 if x > 0.0 else 0.0 for x in deltas]),
        "mean_value_head_delta_pp": mean([row["value_head_delta_pp"] for row in selected]),
    }


def search_policies(train_rows: list[dict], test_rows: list[dict], min_coverage: int) -> list[dict]:
    policies = []
    actions = sorted({row["action"] for row in train_rows})
    for action in actions:
        for threshold in candidate_thresholds([row for row in train_rows if row["action"] == action]):
            train_eval = evaluate_policy(train_rows, action, threshold)
            if train_eval["coverage"] < min_coverage:
                continue
            test_eval = evaluate_policy(test_rows, action, threshold)
            policies.append(
                {
                    "action": action,
                    "threshold_value_head_delta_pp": threshold,
                    "train": train_eval,
                    "test": test_eval,
                }
            )
    return sorted(
        policies,
        key=lambda item: (
            item["test"]["rollout_lcb90_pp"],
            item["test"]["mean_rollout_delta_pp"],
            item["test"]["positive_rate"],
            item["test"]["coverage"],
        ),
        reverse=True,
    )


def make_recommendation(rows: list[dict], policies: list[dict]) -> dict:
    viable = []
    for policy in policies:
        test = policy["test"]
        train = policy["train"]
        if (
            train["coverage"] >= 5
            and test["coverage"] >= 3
            and train["rollout_lcb90_pp"] > 0.0
            and test["mean_rollout_delta_pp"] >= 1.0
            and test["positive_rate"] >= 0.6
        ):
            viable.append(policy)

    if viable:
        best = viable[0]
        return {
            "status": "candidate_policy_found",
            "decision": "Use this intervention gate only as a validation candidate, then confirm with fresh paired rollouts.",
            "policy": {
                "action": best["action"],
                "min_value_head_delta_pp": best["threshold_value_head_delta_pp"],
                "test_mean_rollout_delta_pp": best["test"]["mean_rollout_delta_pp"],
                "test_positive_rate": best["test"]["positive_rate"],
                "test_coverage": best["test"]["coverage"],
            },
        }

    return {
        "status": "no_deployable_policy_yet",
        "decision": "Default to NOOP for Phase 2. The current value-head scores are not calibrated enough to choose interventions.",
        "next_experiment": (
            "When the StatsBomb mount returns, create broader paired rollout labels at a lower value-head threshold, "
            "then train a calibrated scorer on rollout deltas rather than trusting the value head directly."
        ),
        "label_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline calibration scorer for possession intervention labels.")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/intervention_scorer_summary.json")
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--min-train-coverage", type=int, default=4)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    raw_rows = load_rows(paths)
    rows = dedupe_rows(raw_rows)
    for row in rows:
        row["split"] = stable_split(row["match"], args.test_fraction)

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    policies = search_policies(train_rows, test_rows, args.min_train_coverage)

    output = {
        "inputs": [str(path) for path in paths],
        "raw_label_rows": len(raw_rows),
        "deduped_match_action_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "overall": summarize_rows(rows),
        "train": summarize_rows(train_rows),
        "test": summarize_rows(test_rows),
        "top_threshold_policies": policies[:20],
        "recommendation": make_recommendation(rows, policies),
        "rows": rows,
    }
    save_json(output, args.out_json)

    print(f"saved intervention scorer report to {args.out_json}")
    print(json.dumps({
        "deduped_rows": output["deduped_match_action_rows"],
        "train_rows": output["train_rows"],
        "test_rows": output["test_rows"],
        "recommendation": output["recommendation"],
        "top_policy": output["top_threshold_policies"][0] if output["top_threshold_policies"] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
