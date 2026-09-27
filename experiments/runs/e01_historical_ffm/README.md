# E01 historical FFM Table 1 row

Recompute directly from the raw per-match CSV artifacts:

```bash
/home/peter/miniconda3/envs/fb-env/bin/python -m paper_reproduction.recompute_historical_e01
```

The original `model_v3/mad_summary.py` sorts 600 result directories and uses
the slice `[70:520]`, yielding exactly N=450. The independent recomputation
requires all three CSVs for every selected match and records a combined
SHA-256 digest over the 1,350 selected files.

| Metric | Recomputed full precision | Paper |
|---|---:|---:|
| Possession MAD (pp) | 11.9507219 | 11.95 |
| Pass MAD (avg home/away) | 71.2722222 | 71.27 |
| Shot MAD (avg home/away) | 4.3400000 | 4.34 |
| Dribble MAD (avg home/away) | 5.5144444 | 5.52 |
| Total Event MAD | 11.0428758 | 11.04 |

Every manuscript value is recovered at its printed precision.

`Total Event MAD` is the mean per-event-category MAD across home and away,
not error in the aggregate number of match events. This historical definition
is retained because it is the quantity used to produce the paper's 11.04.
