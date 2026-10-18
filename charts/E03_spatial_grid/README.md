# E03 spatial-grid figure

`build_chart.py` reads the recovered native ground-truth and simulated event
matrices from `paper_reproduction/runs/e03_spatial_grid/results.json` and
renders them on football pitches at 4x6, 8x12, and 16x24 resolution. Each
column shares one color scale between its ground-truth and simulated panel.
The reported Zone MAD values are the recovered experimental results.
