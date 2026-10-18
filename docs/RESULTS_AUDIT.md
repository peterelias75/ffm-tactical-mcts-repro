# Results and Figure Audit

This audit records which artifacts were recomputed, which were recovered from
historical runs, and which machine-readable results feed each published plot.

## Audit conclusion

All current chart generators read the saved result JSONs directly, and all
policy result files used in Figures 6–8 and E10 are complete. The figures were
regenerated successfully from those files. E01–E04 are historical artifact
recoveries rather than newly executed clean reruns; this distinction must remain
explicit in the manuscript.

| Experiment | Evidence status | Sample size | Verified result used in plots |
|---|---|---:|---|
| E01 | Historical aggregate recovered | 450 | Possession MAE 11.95 pp; passes 71.27; shots 4.34; dribbles 5.51 |
| E02 | Historical variants recovered | 600 each | Typed augmentation is lowest on the four plotted error metrics |
| E03 | Historical matrices recovered | 595 coarse; 600 regular/fine | Zone MAD 40.13 → 11.26 → 3.07 |
| E04 | Complete historical trace parsed | 600 × 10 runs | Variance 13.87 pp²; absolute error 12.02 pp |
| E05 | Controlled/capped evaluation complete | 200 per policy | Random +1.02; Markov +1.07; plain MCTS +1.36; priors MCTS +0.88 pp |
| E06 | Adapted plain-MCTS sweep complete | 50 per setting | 1-minute decisions best mean gain: +2.57 pp |
| E07 | Adapted plain-MCTS sweep complete | 50 per setting | 5 simulations best mean gain: +2.87 pp |
| E08 | Adapted plain-MCTS sweep complete | 50 per setting | 200 tokens best mean gain: +2.34 pp; 64% improved |
| Figure 7 | Context replay complete | 50 | +18.10 / +6.81 / −5.34 pp by real-possession bucket |
| Figure 8 | Objective comparison complete | 50 per objective | Shots-only: +0.38 shots, +3.98 pp possession; balanced: +0.18, +2.94 pp |
| E10 | Cooldown comparison complete | 50 per setting | 5 minutes best: +3.42 pp; 70% improved; LCB90 +1.80 pp |

## Important interpretation notes

- E05 adapted controlled results do **not** reproduce the manuscript’s original
  priors-guided ranking; plain MCTS is strongest in the controlled rerun.
- E03’s coarse historical aggregation reports 595 matches because the surviving
  historical script skipped five; regular and fine use 600.
- Figure 7 reports coach-minus-real possession, whereas E06–E10 report policy
  minus paired simulated no-action possession. These quantities are not
  interchangeable.
- Figure 8 shot results use the paired simulated baseline because the simulator
  underproduces absolute shot counts.
- E10 is simulation evidence, not a causal estimate of real coaching effects.

## Reproduction checks

```bash
python -m py_compile src/impactful_possession_pipeline/*.py experiments/*.py
python scripts/summarize_results.py
python charts/build_all.py
```

The final separated Figure 6–8 panels and both E10 panels are 1870 × 1320
pixels. E03 is intentionally wider because it contains six pitch panels and
three native-resolution color scales.
