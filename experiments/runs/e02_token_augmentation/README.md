# E02 - Token-augmentation ablation (Figure 3)

Consolidation of historical evaluation artifacts. **No training or evaluation
was run in this task** - this directory only reads and reorganizes results
that already existed under `/home/peter/fb-proj/gpt_dist/`.

## Provenance

Three DistilGPT-2 (~82M param) checkpoints live under `gpt_dist/`, one per
tokenization scheme:

| Paper label | `gpt_dist/` subdir | Checkpoint MD5 |
|---|---|---|
| No-Aug | `gpt_no_subs` | `95cbcb049648088400682d0bfd13594f` |
| Untyped-Aug | `gpt_untagged_subs` | `eff2a31539c12d20f05261be7c0041a8` |
| Typed-Aug | `gpt_tagged_subs` | `7e5362fc0a96ab90d113469b8076b96e` |

**These are confirmed to be three genuinely distinct trained models**, not one
run mislabeled three times: the `model.safetensors` MD5 checksums differ
across all three, and the training scripts differ exactly as the scheme names
imply:

- `gpt_no_subs.py` filters substitution and tactical-shift events out before
  tokenization entirely (`event_to_str` returns `None` for those types, and a
  list-comprehension drops them from the event stream).
- `gpt_untagged.py` keeps those events but stringifies them as generic
  `f"sub - {team}"` / `f"tactical shift - {team}"` tokens (no directionality).
- `gpt_tagged.py` stringifies them via `token_for_substitution` /
  `token_for_tactical_shift`, which rank incoming vs. outgoing player role
  (defender=1 < midfielder=2 < attacker=3) or pre/post formation
  attack-defense balance, and tag the token `attacking` / `defending` /
  `neutral` accordingly.

Each variant was evaluated on the same **n=600 held-out matches** (600
`<match>_results/` directories under each variant's `results/`), using each
variant's own copy of `mad_summary.py` (the three copies are functionally
identical - they differ only in cosmetic docstring wording, confirmed by
diff).

## What `mad_summary.py` actually reports

Read directly from `/home/peter/fb-proj/gpt_dist/gpt_no_subs/mad_summary.py`
(all three copies match). Per match it parses `events.csv`, `positions.csv`,
`zones.csv` and aggregates across all matches into:

- Per-event-type MAD (Home and Away separately), for **every** StatsBomb
  event category present (pass, shot, dribble, carry, pressure, duel, ...
  ~33 categories) - not just shots/dribbles.
- Per-position MAD (Home/Away).
- An 8x12 zone-count MAD matrix (`Total Zone MAD` = its mean).
- `Ball Possession MAD%` - MAD of home ball-touch-event share.
- A convenience block singling out Pass/Shot/Dribble MAD, Home and Away.
- `Total Event MAD` = mean of **all** per-category MADs (Home+Away combined).
  This is **not** a MAD of aggregate event-count totals - there is no
  "total events" row in the source `events.csv` files, so this
  category-averaged figure is the closest available proxy and is reported as
  `total_events_mad` with that caveat spelled out in `results.json`.

There is no separate script output for "shots" or "dribbles" as a single
number; `results.json` averages the Home and Away MAD values for the `shot`
and `dribble` event categories to produce single `shots_mad`/`dribbles_mad`
figures.

## Results summary

| Metric | No-Aug | Untyped-Aug | Typed-Aug | Best |
|---|---:|---:|---:|---|
| Possession MAD (pp) | 13.48 | 13.26 | **12.85** | Typed-Aug |
| Shots MAD (avg H/A) | 7.34 | 4.86 | **4.36** | Typed-Aug |
| Dribbles MAD (avg H/A) | 5.295 | 5.06 | **4.835** | Typed-Aug |
| Total Event MAD (category-avg) | 10.10 | 10.60 | **9.65** | Typed-Aug |

Full per-category numbers, checkpoint hashes, and source file paths are in
`results.json`.

## Does the data support the paper's claim?

**Yes.** The paper's claim for Figure 3 is that Typed-Aug gives the lowest
MAD across all four reported metrics (possession, shots, dribbles,
total_events). Typed-Aug is strictly the best (lowest-MAD) variant on all
four metrics measured here.

One nuance worth flagging: the ordering is **not monotonic** through
Untyped-Aug on every metric. Untyped-Aug beats No-Aug on possession, shots,
and dribbles, but is *worse* than No-Aug on `total_events_mad` (10.60 vs
10.10) and on `total_position_mad` (23.08 vs 22.70, not one of the four
required metrics but visible in `results.json`). So "more augmentation
detail is monotonically better" doesn't hold cleanly at every intermediate
step - only the two-endpoint comparison (No-Aug worst-ish, Typed-Aug best)
that the paper actually claims is fully supported by this data.

## Gaps

None for the four requested metrics - possession, shots, dribbles, and
(the category-averaged proxy for) total_events were all available for all
three variants at full n=600. The only caveat is the `total_events_mad`
definition mismatch noted above and in `results.json`.
