#!/usr/bin/env python3
"""Summarize intervention types and timing in an MCTS results JSON."""

import argparse
import collections
import json
import statistics


def category(action: str) -> str:
    if action.endswith("NOOP"):
        return "noop"
    if "SUB" in action:
        return "substitution"
    return "tactical"


def mean(values):
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    args = parser.parse_args()
    with open(args.results_json) as handle:
        data = json.load(handle)
    rows = data["matches"]
    events = [
        (row["match"], row["delta_pp"], event["minute"], event["chosen_action"])
        for row in rows
        for event in row["interventions"]
    ]

    category_counts = collections.Counter(category(action) for _, _, _, action in events)
    action_counts = collections.Counter(action for _, _, _, action in events)
    by_minute = {
        minute: collections.Counter(
            category(action) for _, _, event_minute, action in events if event_minute == minute
        )
        for minute in range(45, 90)
    }
    active_counts = []
    last_active_minutes = []
    quota_counts = collections.Counter()
    first_action_deltas = collections.defaultdict(list)
    event_weighted_deltas = collections.defaultdict(list)
    match_presence_deltas = collections.defaultdict(list)

    for row in rows:
        interventions = row["interventions"]
        active = [event for event in interventions if category(event["chosen_action"]) != "noop"]
        active_counts.append(len(active))
        last_active_minutes.append(max((event["minute"] for event in active), default=None))
        quota_counts[(row["used_subs"], row["used_tactical_shifts"])] += 1
        first_action_deltas[category(interventions[0]["chosen_action"])].append(row["delta_pp"])
        for action in {event["chosen_action"] for event in active}:
            match_presence_deltas[action].append(row["delta_pp"])

    for _, delta, _, action in events:
        if category(action) != "noop":
            event_weighted_deltas[action].append(delta)

    output = {
        "source_summary": data["summary"],
        "n_matches": len(rows),
        "n_decisions": len(events),
        "category_counts": dict(category_counts),
        "active_actions_per_match": {
            "distribution": dict(collections.Counter(active_counts)),
            "mean": mean(active_counts),
        },
        "quota_distribution_subs_tactics": {
            f"{subs},{tactics}": count for (subs, tactics), count in quota_counts.items()
        },
        "last_active_minute_distribution": dict(collections.Counter(last_active_minutes)),
        "top_actions": action_counts.most_common(),
        "by_minute": {str(minute): dict(counts) for minute, counts in by_minute.items()},
        "first_action_category_outcomes": {
            name: {"n_matches": len(values), "mean_delta_pp": mean(values)}
            for name, values in first_action_deltas.items()
        },
        "action_match_presence_outcomes": sorted(
            (
                {"action": action, "n_matches": len(values), "mean_delta_pp": mean(values)}
                for action, values in match_presence_deltas.items()
            ),
            key=lambda row: (-row["n_matches"], row["action"]),
        ),
        "caveat": (
            "Outcome associations are descriptive and match-level, not causal action effects; "
            "actions were selected jointly by MCTS and multiple actions occur in each match."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
