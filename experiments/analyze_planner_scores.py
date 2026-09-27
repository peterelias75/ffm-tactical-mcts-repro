"""Summarize deterministic value-head action scores from planner JSON."""

import argparse
from collections import defaultdict

from impactful_possession_pipeline.utils import load_json, mean, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    payload = load_json(args.input)
    decisions = []
    action_deltas = defaultdict(list)
    for match in payload.get("matches", []):
        for decision in match.get("interventions", []):
            non_noop = [s for s in decision["scores"] if s["action"] != "ACTION::NOOP"]
            best = max(non_noop, key=lambda s: s["delta"])
            decisions.append({"match": match["match"], "minute": decision["minute"], **best})
            for score in non_noop:
                action_deltas[score["action"]].append(score["delta"] * 100.0)
    best_deltas = [d["delta"] * 100.0 for d in decisions]
    best_utilities = [d["utility"] * 100.0 for d in decisions]
    summary = {
        "num_matches": len(payload.get("matches", [])),
        "num_decisions": len(decisions),
        "fraction_decisions_any_positive_delta": mean([d > 0 for d in best_deltas]),
        "fraction_decisions_clearing_1_5pp_gate": mean([d >= 1.5 for d in best_deltas]),
        "fraction_decisions_positive_utility": mean([u > 0 for u in best_utilities]),
        "mean_best_action_delta_pp": mean(best_deltas),
        "max_best_action_delta_pp": max(best_deltas) if best_deltas else 0.0,
        "mean_best_action_utility_pp": mean(best_utilities),
        "diagnosis": "A candidate must clear both the delta gate and positive utility. Zero positive utility means the configured uncertainty penalty prevents all rollout confirmation.",
    }
    per_action = {
        action: {"n": len(values), "mean_delta_pp": mean(values), "max_delta_pp": max(values)}
        for action, values in sorted(action_deltas.items())
    }
    save_json({"summary": summary, "per_action": per_action, "best_per_decision": decisions}, args.out_json)
    print(summary)


if __name__ == "__main__":
    main()
