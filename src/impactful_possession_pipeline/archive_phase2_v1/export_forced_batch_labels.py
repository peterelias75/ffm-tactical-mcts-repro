import argparse
import json
import math
from pathlib import Path

from .utils import mean, save_json, std


def sem_lcb(values: list[float], z: float = 1.64) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return mean(values) - z * std(values) / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert forced sequential batch results into paired-label rows.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.input_json).read_text())
    if not data.get("complete") and not args.allow_incomplete:
        raise RuntimeError(f"input is not complete: {args.input_json}")

    matches = []
    for result in data.get("results", []):
        deltas = [float(row["delta_pp"]) for row in result.get("rows", [])]
        if not deltas:
            continue
        noop_values = [float(row["noop_possession"]) for row in result.get("rows", [])]
        action_values = [float(row["policy_possession"]) for row in result.get("rows", [])]
        matches.append(
            {
                "match": Path(result["match_json"]).name,
                "match_json": result["match_json"],
                "action": data.get("action", result.get("actions", ["ACTION::NOOP"])[0]),
                "cut_minute": result.get("cut_minute", data.get("cut_minute")),
                "end_minute": result.get("end_minute", data.get("end_minute")),
                "value_head_delta_pp": 0.0,
                "mean_noop_home_possession": mean(noop_values),
                "mean_action_home_possession": mean(action_values),
                "mean_rollout_delta_pp": mean(deltas),
                "std_rollout_delta_pp": std(deltas),
                'lcb90_rollout_delta_pp': sem_lcb(deltas),
                "positive_seed_rate": mean([1.0 if value > 0.0 else 0.0 for value in deltas]),
                "seed_deltas_pp": deltas,
                "n_seeds": len(deltas),
            }
        )

    output = {
        "source": args.input_json,
        "complete": bool(data.get("complete")),
        "action": data.get("action"),
        "matches": matches,
        "summary": {
            "n_matches": len(matches),
            "mean_rollout_delta_pp": mean([row["mean_rollout_delta_pp"] for row in matches]),
            "positive_match_rate": mean([1.0 if row["mean_rollout_delta_pp"] > 0.0 else 0.0 for row in matches]),
            "mean_positive_seed_rate": mean([row["positive_seed_rate"] for row in matches]),
            'mean_lcb90_rollout_delta_pp': mean([row['lcb90_rollout_delta_pp'] for row in matches]),
        },
    }
    save_json(output, args.out_json)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
