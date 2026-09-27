#!/usr/bin/env python3
"""Print the headline metrics from the committed best plain-MCTS result."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/runs/E08_plain_mcts_step1_sim5_n50/lookahead_tokens_200/results.json"


def main() -> None:
    data = json.loads(RESULT.read_text())
    summary = data["summary"]
    fields = (
        "num_matches",
        "mean_delta_pp",
        "pct_matches_improved",
        "mean_gain_when_improved_pp",
        "bootstrap_lcb90_pp",
        "complete",
    )
    print(f"source={RESULT.relative_to(ROOT)}")
    for field in fields:
        print(f"{field}={summary[field]}")


if __name__ == "__main__":
    main()
