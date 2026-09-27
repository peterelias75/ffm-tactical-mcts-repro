# Football Forward Model Tactical Planning

Reproducibility repository for a structured Football Forward Model (FFM) and
Monte Carlo Tree Search (MCTS) tactical intervention experiments on StatsBomb
event data.

The repository contains runnable model/planner code, fixed holdout manifests,
experiment launchers, and the machine-readable outputs used in the paper
reproduction. Large model checkpoints and the upstream StatsBomb Open Data are
not committed; their expected locations and regeneration paths are documented.

## Main finding

For the adapted plain-MCTS experiments, the strongest tested configuration was:

| Setting | Value |
|---|---:|
| Policy | Plain MCTS (uniform priors) |
| Decision interval | 1 minute |
| Simulations per decision | 5 |
| Candidate rollout budget | 200 tokens |
| Evaluation set | 50 held-out matches |

This configuration achieved **+2.34 possession percentage points** versus the
paired no-action simulation, improved **64%** of matches, and had a positive
90% bootstrap lower confidence bound (**+1.08 pp**).

## Repository map

```text
.
├── src/impactful_possession_pipeline/  # model, decoder, planners, evaluators
├── experiments/                        # sweep/analysis scripts and saved runs
│   └── runs/                           # immutable JSON/CSV/PNG result artifacts
├── data/holdouts/                      # match-ID manifests (no raw event data)
├── docs/                               # architecture, catalog, data, reproduction
└── scripts/                            # fast validation entry points
```

Start with:

- [Architecture](docs/ARCHITECTURE.md)
- [Experiment catalog](docs/EXPERIMENTS.md)
- [Fast reproduction guide](docs/REPRODUCE.md)
- [Data and artifact policy](docs/DATA.md)

## Quick validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python -m py_compile src/impactful_possession_pipeline/*.py experiments/*.py
```

A one-match GPU smoke test and the exact full commands are in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Reproducibility conventions

- Every policy comparison uses fixed match IDs and deterministic per-match seeds.
- No-action and policy rollouts are paired.
- Runs write one match at a time and support `--resume`.
- GPU memory is capped in the evaluator.
- Reported gains are percentage-point differences, not relative percentages.
- Historical, adapted, incomplete, and smoke runs remain separated by directory.

## Current status

Experiments E01–E09 and Figure 8 are complete. E10 evaluates active-action
cooldowns of 1, 5, and 10 minutes using the best plain-MCTS configuration; its
5- and 10-minute conditions may appear as resumable partial outputs until the
ongoing run completes.
