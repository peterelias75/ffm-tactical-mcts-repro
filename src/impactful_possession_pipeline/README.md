# Impactful Possession Pipeline

Clean, new implementation for the two-phase project:

1. Train a leakage-free forward simulator from StatsBomb event JSON.
2. Use that simulator for paired counterfactual home-team intervention planning.

The old experiment folders are not touched.

## Why This Version Is Different

- Uses random live cut minutes, not only halftime.
- Trains value heads from prefix-only context, avoiding second-half leakage.
- Represents events as structured tokens instead of fragile free text strings.
- Grounds substitutions and tactical shifts as real timeline action tokens.
- Uses grammar-constrained decoding for valid event-frame generation.
- Plans interventions by paired no-op vs action rollouts under identical seeds.
- Penalizes risk, uncertainty, and intervention cost.

## Folder Map

- `config.py` - all data/model/training/decoding/planning settings.
- `vocabulary.py` - structured token vocabulary.
- `statsbomb_features.py` - StatsBomb parsing, roles, formations, real action grounding.
- `dataset.py` - random-cut future-prediction dataset.
- `model.py` - causal simulator with possession/count/uncertainty heads.
- `decode.py` - grammar-constrained continuation generator.
- `planner.py` - paired counterfactual intervention scorer.
- `train_forward_simulator.py` - phase 1 training command.
- `simulate_continuation.py` - single-match simulation command.
- `plan_interventions.py` - phase 2 intervention planning command.
- `evaluate_pipeline.py` - held-out simulator/planner evaluation.

## Train

```bash
python3 -m impactful_possession_pipeline.train_forward_simulator \
  --events-dir /path/to/statsbomb/data/events \
  --lineups-dir /path/to/statsbomb/data/lineups \
  --out-dir impactful_possession_pipeline/runs/forward_simulator \
  --max-matches 3400 \
  --epochs 30
```

Best checkpoint:

```text
impactful_possession_pipeline/runs/forward_simulator/best_forward_simulator.pt
```

## Simulate One Match

```bash
python3 -m impactful_possession_pipeline.simulate_continuation \
  --checkpoint impactful_possession_pipeline/runs/forward_simulator/best_forward_simulator.pt \
  --match-json /path/to/events/123.json \
  --lineup-json /path/to/lineups/123.json \
  --cut-minute 45 \
  --end-minute 90
```

## Plan Interventions

```bash
python3 -m impactful_possession_pipeline.plan_interventions \
  --checkpoint impactful_possession_pipeline/runs/forward_simulator/best_forward_simulator.pt \
  --match-json /path/to/events/123.json \
  --lineup-json /path/to/lineups/123.json \
  --rollout-seeds 12
```

## Evaluate

Simulator only:

```bash
python3 -m impactful_possession_pipeline.evaluate_pipeline \
  --checkpoint impactful_possession_pipeline/runs/forward_simulator/best_forward_simulator.pt \
  --events-dir /path/to/statsbomb/data/events \
  --lineups-dir /path/to/statsbomb/data/lineups \
  --max-matches 100
```

Simulator plus planner:

```bash
python3 -m impactful_possession_pipeline.evaluate_pipeline \
  --checkpoint impactful_possession_pipeline/runs/forward_simulator/best_forward_simulator.pt \
  --events-dir /path/to/statsbomb/data/events \
  --lineups-dir /path/to/statsbomb/data/lineups \
  --max-matches 50 \
  --run-planner
```

## First Accuracy Target

Treat the first milestone as:

```text
held-out no-intervention second-half possession MAE <= 5 percentage points
```

Do not increase planner complexity until the simulator gets close to that range.
