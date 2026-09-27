# Data and artifacts

## Upstream data

The pipeline expects the official StatsBomb Open Data directory layout:

```text
data/
├── events/<match_id>.json
└── lineups/<match_id>.json
```

Obtain it from the official `statsbomb/open-data` repository and comply with its
attribution terms. Raw event and lineup JSON files are intentionally not copied
into this repository.

## Included data

- Fixed match-ID holdout manifests in `data/holdouts/`.
- Per-match experimental outputs and aggregate summaries in `experiments/runs/`.
- Small derived tables and figures produced by the experiment scripts.

These are sufficient to audit reported metrics without rerunning the model.

## Excluded artifacts

- PyTorch checkpoints (`*.pt`, `*.pth`, `*.ckpt`).
- Raw StatsBomb event/lineup files.
- process IDs, memory traces, and verbose runtime logs.
- caches and temporary partial downloads.

The main evaluator expects the value-head checkpoint used by the experiments.
Pass its local path explicitly with `--checkpoint`. To rebuild it, follow the
training entry points in `src/impactful_possession_pipeline/`.
