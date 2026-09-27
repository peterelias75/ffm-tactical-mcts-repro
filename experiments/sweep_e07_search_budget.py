#!/usr/bin/env python3
"""E07 sweep: search budget (--num-simulations in {1, 5, 10, 30, 50}), --step-minutes 5 fixed.

Also captures wall-clock latency per intervention decision, since
evaluate_mcts_pipeline.py has no built-in per-decision timer.

Latency instrumentation -- read before trusting the numbers
-------------------------------------------------------------
This sweep times each condition's *whole* subprocess call (wall clock) and
divides by (newly-processed matches x decisions-per-match), where
decisions-per-match = len(range(45, 90, step_minutes)) = 9 for
step_minutes=5. This is the "simple approach" explicitly sanctioned by the
task that requested this script, but it is a CONFLATED estimate, not a clean
isolation of MCTS search cost. Per match, the timed work is:

    noop_rollout_cost                              (1x, fixed, independent of num_simulations)
  + sum over 9 decisions of (
        num_simulations * per_candidate_rollout_cost   (scales with num_simulations)
      + advance_generation_cost                          (1x per decision, fixed)
    )

So elapsed_per_decision ~= noop_rollout_cost/9 + num_simulations * rollout_cost
+ advance_generation_cost. The first and third terms are fixed overhead that
does NOT scale with num_simulations, so the *absolute* seconds-per-decision
figure will read higher than the paper's isolated per-decision search cost,
especially at low num_simulations (e.g. num_simulations=1) where fixed
overhead dominates. Checkpoint-load time per subprocess invocation is also
folded in, amortized over however many matches that invocation newly
processes -- material for small --max-matches smoke tests, negligible at
full holdout scale. The *relative* scaling across num_simulations should
still be directly informative even though the absolute numbers are an
overestimate. See timing.json in each condition directory for the raw
elapsed/matches-processed numbers this is derived from.

Every other MCTS knob is left at evaluate_mcts_pipeline.py's CLI defaults
(uniform_prior_weight=0.05, i.e. priors-guided MCTS) except step_minutes,
which is pinned to 5 for every condition per the paper's E07 setup.

Usage
-----
Smoke test:
    python sweep_e07_search_budget.py --max-matches 2 \\
        --run-root /home/peter/fb-proj/paper_reproduction/runs/smoke_e07 --gpu 0

Full paper-scale run (run under screen):
    screen -dmS sweep_e07 /home/peter/miniconda3/envs/fb-env/bin/python \\
        -m paper_reproduction.sweep_e07_search_budget --gpu 0
"""

import argparse
from pathlib import Path

from .sweep_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HOLDOUT,
    DEFAULT_SEED,
    REPO_ROOT,
    decisions_per_match,
    holdout_match_count,
    run_condition,
    save_json,
)

NUM_SIMULATIONS_VALUES = [1, 5, 10, 30, 50]
STEP_MINUTES = 5
ROLLOUT_TOKENS = 250


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(REPO_ROOT / "paper_reproduction/runs/E07_search_budget"))
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--holdout-json", default=DEFAULT_HOLDOUT)
    ap.add_argument("--noop-cache-json", default=None, help="Completed compatible policy results used to reuse deterministic no-intervention rollouts.")
    ap.add_argument("--gpu", type=int, choices=[0, 1, 2], default=0)
    ap.add_argument("--cut-minute", type=int, default=45)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    expected = holdout_match_count(args.holdout_json, args.max_matches)
    decisions = decisions_per_match(STEP_MINUTES, cut_minute=args.cut_minute)

    results = []
    for num_sim in NUM_SIMULATIONS_VALUES:
        label = f"num_simulations_{num_sim:02d}"
        cond_dir = run_root / label
        out_json = cond_dir / "results.json"
        cli_args = [
            "--checkpoint", args.checkpoint,
            "--holdout-json", args.holdout_json,
            "--cut-minute", str(args.cut_minute),
            "--step-minutes", str(STEP_MINUTES),
            "--num-simulations", str(num_sim),
            "--rollout-lookahead-tokens", str(ROLLOUT_TOKENS),
            "--seed", str(args.seed),
            "--resume",
            "--out-json", str(out_json),
        ]
        if args.noop_cache_json:
            cli_args += ["--noop-cache-json", args.noop_cache_json]
        if args.max_matches:
            cli_args += ["--max-matches", str(args.max_matches)]
        res = run_condition(
            cond_dir, "impactful_possession_pipeline.evaluate_mcts_pipeline", cli_args,
            args.gpu, out_json, expected, label,
        )
        newly = res["newly_processed"]
        elapsed = res["elapsed_seconds"]
        avg_latency = (elapsed / (newly * decisions)) if (newly and elapsed is not None) else None
        results.append({"num_simulations": num_sim, "avg_latency_seconds_per_decision": avg_latency, **res})

    summary_rows = [
        {
            "num_simulations": r["num_simulations"],
            "num_matches": r["summary"].get("num_matches"),
            "mean_delta_pp": r["summary"].get("mean_delta_pp"),
            "pct_matches_improved": r["summary"].get("pct_matches_improved"),
            "mean_gain_when_improved_pp": r["summary"].get("mean_gain_when_improved_pp"),
            "bootstrap_lcb90_pp": r["summary"].get("bootstrap_lcb90_pp"),
            "avg_latency_seconds_per_decision": r["avg_latency_seconds_per_decision"],
            "elapsed_seconds_this_call": r["elapsed_seconds"],
            "newly_processed_matches_this_call": r["newly_processed"],
            "complete": r["summary"].get("complete"),
        }
        for r in results
    ]
    save_json(run_root / "summary.json", {
        "experiment": "E07_search_budget",
        "run_root": str(run_root),
        "max_matches_requested": args.max_matches,
        "step_minutes": STEP_MINUTES,
        "rollout_tokens": ROLLOUT_TOKENS,
        "decisions_per_match": decisions,
        "latency_caveat": (
            "avg_latency_seconds_per_decision conflates fixed per-match overhead "
            "(one no-intervention reference rollout, plus one match-advancing "
            "generation per decision) with the num_simulations-scaling MCTS search "
            "cost -- see this script's module docstring. Treat absolute values as "
            "an overestimate; relative scaling across num_simulations should still "
            "be informative."
        ),
        "paper_target": [
            {"iterations": 1, "pct_improved": 38.0, "latency_seconds": 3.0},
            {"iterations": 50, "pct_improved": 82.0, "latency_seconds": 180.0},
        ],
        "conditions": summary_rows,
    })
    print("\nE07 sweep summary:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
