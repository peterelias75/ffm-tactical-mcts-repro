# E05 historical policy-log audit

This package parses the surviving `gpt_medium` stdout logs for random,
classic-Markov, plain-MCTS, and priors-guided policies. It does **not** treat
them as a controlled reproduction because the scripts evaluated inconsistent
match populations and only printed possession to two decimals.

Recreate the structured result:

```bash
/home/peter/miniconda3/envs/fb-env/bin/python -m paper_reproduction.parse_historical_policies \
  --out-json paper_reproduction/runs/e05_historical_policy_audit/results.json
```

The JSON records SHA-256 hashes of every source script/log, every parsed row,
first-200 summaries, full-log summaries, unique/duplicate match diagnostics,
and two explicit gain definitions.

## First 200 rows: direct possession gain

| Policy | Mean gain (pp) | Improved | Gain when improved (pp) | Paper |
|---|---:|---:|---:|---|
| Random | -0.108 | 47.0% | 2.373 | +1.02 / 32.55% / 3.12 |
| Classic Markov | -0.218 | 45.0% | 2.364 | +1.66 / 41.2% / 4.03 |
| Plain MCTS | +0.101 | 52.5% | 4.907 | +3.10 / 52.0% / 6.00 |
| Priors | -0.361 | 44.0% | 2.313 | +4.12 / 61.0% / 9.43 |

The plain-MCTS improved percentage is close to the paper's 52%, but its mean
gain and conditional gain are not. The other policy rows differ materially.
Using reduction in absolute error versus real possession also fails to recover
the complete claimed table; both definitions are preserved in `results.json`.

## Why clean runs are still required

- Random, classic-Markov, and plain-MCTS logs contain 3,433 unique matches.
- The priors log contains 2,964 unique matches despite the currently saved
  script selecting 600, showing that stdout and source were accumulated or
  edited across different executions.
- The paper reports N=200 but does not preserve the exact historical 200 IDs.
- The new controlled suite pins the same 200 leakage-safe checkpoint holdout
  IDs for all four policies and persists full-precision output after each match.

Therefore this package is an honest audit and useful counter-evidence, not a
substitute for `impactful_possession_pipeline.evaluate_*_policy_pipeline`.
