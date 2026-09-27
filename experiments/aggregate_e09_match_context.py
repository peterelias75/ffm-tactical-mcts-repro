#!/usr/bin/env python3
"""E09: match-context stratification. Pure aggregation -- no new sweep.

Buckets a priors-guided-MCTS evaluate_mcts_pipeline.py results.json's
per-match rows by `real_home_possession` into <45%, 45-55%, >55%, and reports
mean `delta_pp` per bucket. Intended input: the "matches" array from any
evaluate_mcts_pipeline.py --out-json run at the paper's default config
(uniform_prior_weight left at its 0.05 default = priors-guided MCTS), e.g.
the step_minutes_05 condition of an E06 sweep, or the num_simulations_10
condition of an E07 sweep, or any dedicated E05-equivalent priors_mcts run.

Accepts one or more --results-json paths; rows are pooled by match id (a
match id repeated across inputs is a warning, and the later file's row wins
-- pass only inputs that are genuinely the same policy/config if you combine
files, otherwise you are averaging across different search settings without
meaning to).

Usage
-----
    python aggregate_e09_match_context.py \\
        --results-json /path/to/E06_intervention_frequency/step_minutes_05/results.json \\
        --out-json /path/to/E09_match_context/results.json
"""

import argparse

from .sweep_common import load_json, save_json

BUCKETS = [("lt45", 0.0, 0.45), ("45to55", 0.45, 0.55), ("gt55", 0.55, 1.0000001)]


def bucket_for(real_possession: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= real_possession < hi:
            return name
    return "gt55"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-json", nargs="+", required=True,
                     help="One or more evaluate_mcts_pipeline.py --out-json files (priors-guided MCTS results).")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows_by_match = {}
    for path in args.results_json:
        data = load_json(path)
        matches = data.get("matches", [])
        if not matches:
            print(f"warning: {path} has no 'matches' rows (complete={data.get('summary', {}).get('complete')})")
        for row in matches:
            match_id = str(row.get("match"))
            if match_id in rows_by_match:
                print(f"warning: duplicate match id {match_id} seen again in {path}; overwriting previous row")
            rows_by_match[match_id] = row

    rows = list(rows_by_match.values())
    if not rows:
        raise SystemExit("no match rows found in any --results-json input; nothing to aggregate")

    buckets = {name: [] for name, _, _ in BUCKETS}
    skipped_no_real_possession = 0
    for row in rows:
        rp = row.get("real_home_possession")
        if rp is None:
            skipped_no_real_possession += 1
            continue
        buckets[bucket_for(rp)].append(row)

    bucket_summary = {}
    for name, lo, hi in BUCKETS:
        deltas = [r["delta_pp"] for r in buckets[name]]
        bucket_summary[name] = {
            "range": [lo, hi],
            "n_matches": len(deltas),
            "mean_delta_pp": mean(deltas),
            "match_ids": [r["match"] for r in buckets[name]],
        }

    out = {
        "experiment": "E09_match_context",
        "inputs": args.results_json,
        "n_total_matches_pooled": len(rows),
        "n_skipped_missing_real_possession": skipped_no_real_possession,
        "paper_target": {"lt45": 8.75, "45to55": 6.68, "gt55": 2.50},
        "buckets": bucket_summary,
    }
    save_json(args.out_json, out)
    print("E09 aggregation:")
    for name, info in bucket_summary.items():
        print(f"  {name}: n={info['n_matches']} mean_delta_pp={info['mean_delta_pp']}")


if __name__ == "__main__":
    main()
