import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .utils import mean, save_json, std


def load_results(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        deltas = [float(row["delta_pp"]) for row in data.get("rows", [])]
        if not deltas:
            continue
        actions = [action for action in data.get("actions", []) if action != "ACTION::NOOP"]
        rows.append(
            {
                "source": str(path),
                "match": Path(str(data.get("match_json", path.stem))).stem,
                "complete": bool(data.get("complete")),
                "actions": data.get("actions", []),
                "non_noop_actions": actions,
                "mean_delta_pp": mean(deltas),
                "median_delta_pp": float(median(deltas)),
                "std_delta_pp": std(deltas),
                "positive_seed_rate": mean([1.0 if delta > 0.0 else 0.0 for delta in deltas]),
                "n_seeds": len(deltas),
                "seed_deltas_pp": deltas,
            }
        )
    return rows


def summarize_group(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_delta_pp": mean(values),
        "median_delta_pp": float(median(values)) if values else 0.0,
        "std_delta_pp": std(values),
        "positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in values]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired validation results for sequential intervention plans.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(sorted(Path().glob(pattern)))
    paths = [path for path in paths if path.is_file()]
    plans = load_results(paths)

    all_seed_deltas = [delta for plan in plans for delta in plan["seed_deltas_pp"]]
    action_seed_deltas: dict[str, list[float]] = defaultdict(list)
    action_plan_deltas: dict[str, list[float]] = defaultdict(list)
    sequence_deltas: dict[str, list[float]] = defaultdict(list)
    action_counts = Counter()

    for plan in plans:
        sequence = " -> ".join(plan["actions"])
        sequence_deltas[sequence].append(plan["mean_delta_pp"])
        for action in plan["non_noop_actions"]:
            action_counts[action] += 1
            action_seed_deltas[action].extend(plan["seed_deltas_pp"])
            action_plan_deltas[action].append(plan["mean_delta_pp"])

    output = {
        "input_files": [str(path) for path in paths],
        "n_plans": len(plans),
        "n_seeds": len(all_seed_deltas),
        "overall": summarize_group(all_seed_deltas),
        "match_positive_rate": mean([1.0 if plan["mean_delta_pp"] > 0.0 else 0.0 for plan in plans]),
        "action_counts": dict(action_counts.most_common()),
        "by_action_seed_level": {
            action: summarize_group(values) for action, values in sorted(action_seed_deltas.items())
        },
        "by_action_plan_level": {
            action: summarize_group(values) for action, values in sorted(action_plan_deltas.items())
        },
        "by_sequence": {
            sequence: summarize_group(values) for sequence, values in sorted(sequence_deltas.items())
        },
        "plans": plans,
        "recommendation": {
            "status": "requires_conservative_rerun",
            "reason": "The paired whole-flow policy is negative versus all-NOOP. Treat raw planner utilities as untrusted until a fresh conservative gate beats NOOP.",
            "next_allowed_actions": [
                "ACTION::HOME_SUB_NEUTRAL",
                "ACTION::HOME_SUB_ROLE_WINGER",
            ],
            "blocked_actions_for_next_pilot": [
                "ACTION::HOME_TACTIC_FORM_343",
                "ACTION::HOME_TACTIC_NEUTRAL",
            ],
        },
    }
    save_json(output, args.out_json)
    print(json.dumps({k: output[k] for k in ["n_plans", "n_seeds", "overall", "match_positive_rate", "recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
