#!/usr/bin/env python3
"""E08 sweep: rollout lookahead (--rollout-lookahead-tokens in {0, 40, 200, 400}),
--step-minutes 5 and --num-simulations 10 fixed.

evaluate_mcts_pipeline.py passes this value as a literal max_new_tokens cap
used only for candidate evaluation. `planner_cfg.step_minutes` (fixed at 5
here) still governs real match advancement. A zero budget evaluates the
injected action token without generating a continuation.

The paper's target numbers are non-monotonic in lookahead tokens (200 tokens
beats 400 tokens on both pct_improved and gain-when-improved) -- if this
sweep reproduces something similar, that is flagged in claims.json/the task
brief as a plausible real finding to report as-is, not evidence of a bug in
this runner or the underlying pipeline.

Usage
-----
Smoke test:
    python sweep_e08_lookahead.py --max-matches 2 \\
        --run-root /home/peter/fb-proj/paper_reproduction/runs/smoke_e08 --gpu 0

Full paper-scale run (run under screen):
    screen -dmS sweep_e08 /home/peter/miniconda3/envs/fb-env/bin/python \\
        -m paper_reproduction.sweep_e08_lookahead --gpu 0
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

LOOKAHEAD_TOKENS_VALUES = [0, 40, 200, 400]
STEP_MINUTES = 5
NUM_SIMULATIONS = 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(REPO_ROOT / "paper_reproduction/runs/E08_lookahead"))
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

    results = []
    for tokens in LOOKAHEAD_TOKENS_VALUES:
        label = f"lookahead_tokens_{tokens:03d}"
        cond_dir = run_root / label
        out_json = cond_dir / "results.json"
        cli_args = [
            "--checkpoint", args.checkpoint,
            "--holdout-json", args.holdout_json,
            "--cut-minute", str(args.cut_minute),
            "--step-minutes", str(STEP_MINUTES),
            "--num-simulations", str(NUM_SIMULATIONS),
            "--rollout-lookahead-tokens", str(tokens),
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
        results.append({"rollout_lookahead_tokens": tokens, **res})

    summary_rows = [
        {
            "rollout_lookahead_tokens": r["rollout_lookahead_tokens"],
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
        "experiment": "E08_lookahead",
        "run_root": str(run_root),
        "max_matches_requested": args.max_matches,
        "step_minutes": STEP_MINUTES,
        "num_simulations": NUM_SIMULATIONS,
        "paper_target": [
            {"lookahead_tokens": 0, "pct_improved": 30.0, "gain_when_improved_pp": 2.67},
            {"lookahead_tokens": 200, "pct_improved": 82.0, "gain_when_improved_pp": 9.87},
            {"lookahead_tokens": 400, "pct_improved": 67.0, "gain_when_improved_pp": 6.22},
        ],
        "conditions": summary_rows,
    })
    print("\nE08 sweep summary:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
