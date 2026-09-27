#!/usr/bin/env python3
"""E06 sweep: intervention frequency (--step-minutes in {1, 5, 10}).

Matched-budget reasoning
------------------------
The paper specifies literal candidate-evaluation budgets of 30/150/300
generated tokens for 1/5/10-minute intervention intervals. This sweep passes
those caps explicitly. The interval still controls only how far the selected
action advances the real simulated match.

Every other MCTS knob (num_simulations, uniform_prior_weight, c_puct,
selection_policy, ...) is left at evaluate_mcts_pipeline.py's own CLI
defaults for every condition -- i.e. the priors-guided MCTS policy
(uniform_prior_weight=0.05) at num_simulations=10, matching the paper's
Table 2 default search config. Only step_minutes varies across conditions.

Usage
-----
Smoke test (2 matches, ~a few minutes to tens of minutes per condition):
    python sweep_e06_intervention_frequency.py --max-matches 2 \\
        --run-root /home/peter/fb-proj/paper_reproduction/runs/smoke_e06 --gpu 0

Full paper-scale run (all matches in the holdout, run under screen):
    screen -dmS sweep_e06 /home/peter/miniconda3/envs/fb-env/bin/python \\
        -m paper_reproduction.sweep_e06_intervention_frequency --gpu 0
(re-invoking the same command later with the same --run-root safely resumes;
already-complete conditions are skipped, in-progress ones resume mid-match-list)
"""

import argparse
from pathlib import Path

from .sweep_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HOLDOUT,
    DEFAULT_SEED,
    REPO_ROOT,
    holdout_match_count,
    run_condition,
    save_json,
)

STEP_MINUTES_TO_ROLLOUT_TOKENS = {1: 30, 5: 150, 10: 300}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(REPO_ROOT / "paper_reproduction/runs/E06_intervention_frequency"))
    ap.add_argument("--max-matches", type=int, default=None, help="Slice the holdout to this many matches (e.g. 2 for a smoke test). Omit for the full holdout.")
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

    results = []
    for step_minutes, rollout_tokens in STEP_MINUTES_TO_ROLLOUT_TOKENS.items():
        label = f"step_minutes_{step_minutes:02d}"
        cond_dir = run_root / label
        out_json = cond_dir / "results.json"
        cli_args = [
            "--checkpoint", args.checkpoint,
            "--holdout-json", args.holdout_json,
            "--cut-minute", str(args.cut_minute),
            "--step-minutes", str(step_minutes),
            "--rollout-lookahead-tokens", str(rollout_tokens),
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
        results.append({"step_minutes": step_minutes, "rollout_tokens": rollout_tokens, **res})

    summary_rows = [
        {
            "step_minutes": r["step_minutes"],
            "rollout_tokens": r["rollout_tokens"],
            "num_matches": r["summary"].get("num_matches"),
            "mean_delta_pp": r["summary"].get("mean_delta_pp"),
            "pct_matches_improved": r["summary"].get("pct_matches_improved"),
            "mean_gain_when_improved_pp": r["summary"].get("mean_gain_when_improved_pp"),
            "bootstrap_lcb90_pp": r["summary"].get("bootstrap_lcb90_pp"),
            "complete": r["summary"].get("complete"),
        }
        for r in results
    ]
    save_json(run_root / "summary.json", {
        "experiment": "E06_intervention_frequency",
        "run_root": str(run_root),
        "max_matches_requested": args.max_matches,
        "paper_target": {"interval_minutes": 5, "pct_improved": 72.0},
        "conditions": summary_rows,
    })
    print("\nE06 sweep summary:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
