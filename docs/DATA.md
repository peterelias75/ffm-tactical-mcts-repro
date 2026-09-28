# Data, splits, and artifact provenance

## Upstream data

The pipeline expects official StatsBomb Open Data:

```text
data/
├── events/<match_id>.json
└── lineups/<match_id>.json
```

Events provide the chronological match stream. Lineups recover team, player-role, and formation context. Obtain the files from the official `statsbomb/open-data` repository and follow its attribution terms.

Raw events are not duplicated here. Match-ID manifests define every evaluation subset, keeping the code repository small while making selection auditable.

## Evaluation subsets

- `holdout_table2_200.json` is the fixed 200-match policy-comparison holdout.
- Adapted E06–E08 use its first 50 matches for consistent, affordable sweeps.
- Figure 8 uses a recorded set of 50 matches whose real second-half home possession is below 45%.
- E10 uses the same 50-match adapted holdout as E06–E08.

A directory with fewer completed rows is not a new split. Selection is defined by the holdout manifest plus `max_matches`.

## Included artifacts

- fixed match-ID manifests in `data/holdouts/`;
- per-match policy/baseline outputs in `experiments/runs/`;
- aggregate summaries and completion flags;
- small derived tables, heatmaps, and audit records;
- scripts used to launch, replay, and aggregate experiments.

Each result JSON normally contains:

- `summary`: checkpoint reference, seed, search/decoder configuration, requested/completed counts, aggregate metrics, and `complete`;
- `matches`: paired baseline/policy values and minute-level intervention traces.

These artifacts allow reported metrics to be audited without rerunning the model.

## Excluded artifacts

- model checkpoints (`*.pt`, `*.pth`, `*.ckpt`);
- raw StatsBomb JSON;
- process IDs, verbose logs, memory traces, caches, and temporary downloads.

Checkpoint exclusion is intentional: weights are large, cannot be meaningfully code-reviewed, and exceed ordinary source-repository limits. Result records preserve the checkpoint path/name. Pass a local checkpoint via `--checkpoint`, or rebuild one using the training entry points in `src/impactful_possession_pipeline/`.

## Leakage and interpretation

Holdouts are pinned before policy comparison. Real second-half outcomes are used for stratification and external description, while the policy’s main causal-style comparison is paired MCTS versus paired no-action simulation. This does not turn the simulator into a real-world causal estimator; it only controls the simulated comparison.
