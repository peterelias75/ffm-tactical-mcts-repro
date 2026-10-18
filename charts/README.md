# Experiment charts

Each subdirectory contains a generated `chart.png` and a local `build_chart.py`.
Builders read the committed JSON artifacts; chart values are never manually copied.

Run one chart from the repository root:

```bash
python charts/E08_lookahead/build_chart.py
```

Rebuild the full suite:

```bash
python charts/build_all.py
```

The shared `chartlib.py` defines consistent typography, colors, labels, source
footnotes, and interim-run marking. E10 is automatically labelled interim until
all cooldown result summaries have `complete: true`.


## Final separated paper panels

Figures 6–8 are stored as independently usable panels, each exactly
`1870 x 1320` pixels with standalone titles and axis labels:

- `Figure6_mcts_sweeps/figure_6a.png` through `figure_6c.png`
- `Figure7_match_context/figure_7a.png` and `figure_7b.png`
- `Figure8_objectives/figure_8a.png` through `figure_8c.png`

E10 contains two final N=50 plots: cooldown performance and active-intervention
timing. Run `python charts/build_all.py` from the repository root to regenerate
all chart assets.
