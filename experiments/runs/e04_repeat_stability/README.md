# E04 repeat-simulation stability

The complete historical stdout at `model_v3/model_v3_mean_var.txt` contains
600 match summaries and all 6,000 individual simulated possession values.
It was produced by `model_v3/model_v3_mean_var.py`, which runs ten stochastic
simulations per held-out match and uses Python's `statistics.variance` (sample
variance, denominator N-1).

Parse and validate it with:

```bash
/home/peter/miniconda3/envs/fb-env/bin/python -m paper_reproduction.parse_historical_e04 \
  --log model_v3/model_v3_mean_var.txt \
  --script model_v3/model_v3_mean_var.py \
  --out-json paper_reproduction/runs/e04_repeat_stability/results.json
```

The parser refuses incomplete input unless it finds exactly 600 summaries,
ten uniquely indexed runs per match, and therefore 6,000 simulations total.
It also records SHA-256 hashes for the source log and script.

## Recovered result

- Mean within-match sample variance: 13.86665 pp² (paper: 13.87)
- Mean absolute error of the ten-run mean: 12.016983 pp (paper: 12.02)

Both claims reproduce exactly at the manuscript's printed precision.

## Rerun caveat

The historical script declares `trained_transformer_model_v3.pt`, but that
exact filename is no longer present in `model_v3/`. The complete immutable
output evidence survives, but a bit-for-bit rerun requires identifying the
original weights. The modern runner `paper_reproduction.run_repeat_stability_shard`
is resumable after every simulation, enforces the GPU memory cap, and reports
both sample and population variance; it operates on the available modern
checkpoint's persisted 340-match holdout rather than silently claiming N=600.
