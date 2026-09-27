# E03 - Spatial-grid ablation (Figure 4)

Consolidation of historical evaluation artifacts. **No training or
evaluation was run in this task** - this directory only reads and
reorganizes results that already existed under `/home/peter/fb-proj/gpt_medium/`
(and cross-checks against `/home/peter/fb-proj/pitch_grid_splits/zones_plt.py`).

## Provenance and grid-resolution mapping

`gpt_medium/` contains four trained-and-evaluated variants: `base`,
`fine-grid`, `small-grid`, and `top-p`. Per the task scope, **`top-p` is a
decoding-strategy ablation and is excluded** - it changes how tokens are
sampled at generation time, not the pitch discretization, so it isn't part
of Figure 4's comparison.

The task asked to verify the resolution mapping rather than trust variant
names, and it's a good thing it did: the script most likely responsible for
generating the "fine-grid" artifacts, `gpt_medium/gpt_fine_grid.py`, is
internally self-contradictory. Its docstring says "Map (x,y) to finer
discrete zones using a 16x24 grid", but its actual bin constants are
`X_BINS=4, Y_BINS=6` (the **coarse** grid's bin count), and its
`SIM_DIR`/`FINETUNED_DIR` path strings reference `"small_grid"`/
`"small-grid"`, not fine-grid. This is consistent with the same
edit-scripts-in-place-between-runs pattern visible in `mad_summary.py`
(its current `RESULTS_DIR`/`OUTPUT_FILE` constants point at the small-grid
run, `[5:]`-slice and all - see below) - these are one-off research scripts
whose in-file comments/paths were not kept in sync with each edit, so their
*current* saved state cannot be trusted to describe which historical run
produced which artifact.

**What we actually used to establish the mapping:** the literal shape of the
Average GT/SIM/DIFF zone matrices as *written to each `mad_summary*.txt`
output file*. That's real output data, not a variable that could have been
edited after the fact:

| `mad_summary*.txt` | Matrix shape (rows x cols) | Cell count | Maps to |
|---|---|---|---|
| `mad_summary.txt` | 8 x 12 | 96 | **regular 8x12** (paper default) |
| `mad_summary_fine_grid.txt` | 24 x 16 | 384 | **fine 16x24** (transposed row/col vs. paper's "16x24" label, same cell count, same 4x-per-axis resolution increase) |
| `mad_summary_small_grid.txt` | 6 x 4 | 24 | **coarse 4x6** |

This was then cross-checked against three independently-named, differently
sized results directories and three distinct model checkpoints:

| Grid | `gpt_medium` variant | Checkpoint dir | `model.safetensors` MD5 | Results dir |
|---|---|---|---|---|
| coarse 4x6 | small-grid | `football-finetuned-gpt2-m-tagged-small-grid` | `bb1296b3e70acafd1088682115ec5675` | `results_small_grid/` (600 dirs) |
| regular 8x12 | base | `football-finetuned-gpt2-m-tagged` | `b72bff0d9a47fcb38c3f330bcc30f463` | `results/` (600 dirs) |
| fine 16x24 | fine-grid | `football-finetuned-gpt2-m-tagged-fine-grid` | `826ae846efb73515cebbc0650484f17f` | `results_fine_grid/` (600 dirs) |

All three checkpoint MD5s differ, and the earlier project-memory note that
"Total Zone MAD ranges 3.07 to 40.13 across variants" checks out exactly at
the fine-grid and coarse-grid ends respectively, which is strong independent
corroboration of this mapping.

**Excluded, abandoned variant:** `football-finetuned-gpt2-m-tagged-fine-grid-small/`
(with training log `gpt_fine_grid_small.txt`, only 35 lines / ~2 eval
checkpoints, no final `model.safetensors`) looks like an abandoned
fourth-grid or fine-tuning-of-fine-grid experiment that never finished. It
has no matching `results_*`/`mad_summary_*` output, so it is not one of the
three E03 variants and is reported here as genuinely missing rather than
guessed at.

## Sample size caveat

All three variants have 600 complete `<match>_results/` directories
available. However, the **small-grid** `mad_summary.py` run specifically
slices `match_dirs[5:]` before aggregating (visible in the currently-saved
script state, whose `RESULTS_DIR='results_small_grid'` / `OUTPUT_FILE=
'mad_summary_small_grid.txt'` constants match that run), so its aggregate
numbers are computed over **595** of the 600 available matches, not 600. No
equivalent skip evidence was found for the base or fine-grid outputs (script
state for those runs was not preserved - it was overwritten in place by
later edits - but their results directories show no similar irregularity).
This is recorded per-variant in `results.json`.

## What's in `results.json`

Per grid resolution: `possession_mad_pct`, `shots_mad` (home/away/avg),
`dribbles_mad` (home/away/avg), `total_events_mad` (same category-averaged
caveat as E02 - see `metric_definitions`), `total_zone_mad`, and the full
**GT / SIM / DIFF zone-count matrices** at native resolution, parsed
programmatically and verbatim from each `mad_summary*.txt`.

### Note on `pitch_grid_splits/zones_plt.py`

That file contains two hardcoded matrix blocks: an 8x12 block (commented
out, matches the base variant) and an active 24x16 block that actually runs
and produces `zone_heatmaps_GT_vs_SIM_24x16_clean.png`. Its active-block
numbers do **not** exactly match `mad_summary_fine_grid.txt`'s Average
GT/SIM matrices cell-for-cell - several cells differ by up to ~8, mostly in
the last column - suggesting it was hand-edited for visualization at some
point rather than being a faithful copy. `results.json` and the plotting
script below use the values parsed directly from `mad_summary*.txt` instead,
since that's the unedited script output. No hardcoded 4x6 matrices existed
anywhere before this task; those were extracted fresh from
`mad_summary_small_grid.txt`.

## Results summary

| Metric | Coarse 4x6 | Regular 8x12 | Fine 16x24 |
|---|---:|---:|---:|
| **Total Zone MAD** (primary spatial metric) | 40.13 | 11.26 | **3.07** |
| Possession MAD (pp) | 12.18 | 11.64 | **11.22** |
| Shots MAD (avg H/A) | 5.585 | 3.35 | **3.185** |
| Dribbles MAD (avg H/A) | 7.20 | 5.21 | **4.52** |
| Total Event MAD (category-avg) | 11.14 | 10.64 | **9.44** |
| Total Position MAD (not a grid metric, for reference) | **22.63** | 23.82 | 24.78 |

## Does the data support the paper's claim?

**Yes.** The paper claims finer grid = lower MAD generally. Total Zone MAD -
the metric most directly tied to spatial discretization - drops
monotonically and dramatically from coarse to fine: **40.13 -> 11.26 -> 3.07**,
better than a 13x improvement end to end. Possession MAD, Shots MAD,
Dribbles MAD, and Total Event MAD all also decrease monotonically from
coarse to fine grid.

The one metric that does **not** follow the trend is Total Position MAD,
where coarse is actually best and fine is worst. This isn't a contradiction
of the paper's claim so much as a scope note: player-position labels aren't
a direct function of pitch-zone bin granularity the way event counts and
zone occupancy are, so there's no strong a priori reason to expect it to
track grid resolution the same way. It's reported for transparency in
`results.json`/`claim_check` but doesn't affect the headline finding.

## Heatmap plotting

See `plot_zone_heatmaps.py` in this directory - a small, reusable script
generalizing the one-off `pitch_grid_splits/zones_plt.py` (which had
matrices hardcoded for one resolution at a time and depended on `mplsoccer`).
It reads GT/SIM matrices straight out of this directory's `results.json` for
any of the three resolutions and renders a GT-vs-SIM heatmap comparison PNG
using only matplotlib (no extra pitch-drawing dependency required). Example
outputs for all three resolutions are saved alongside it:
`zone_heatmap_coarse_4x6.png`, `zone_heatmap_regular_8x12.png`,
`zone_heatmap_fine_16x24.png`.

## Gaps

None of substance. All four requested metric categories (possession, shots,
dribbles, total_events) and the GT/SIM/DIFF zone matrices were available for
all three in-scope grid resolutions. The only real gap is the abandoned
`fine-grid-small` checkpoint noted above, which was never a required part of
this comparison.
