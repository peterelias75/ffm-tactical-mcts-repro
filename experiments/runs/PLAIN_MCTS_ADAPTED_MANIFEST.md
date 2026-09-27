# Adapted plain-MCTS reproduction manifest

## Frozen analysis decision

- Planner: plain MCTS (`uniform_prior_weight=1.0`).
- Selection basis: controlled E05 comparison on the shared ordered holdout.
- Interpretation: E06-E09 are adapted/exploratory plain-MCTS analyses, not
  exact reproductions of the manuscript's priors-guided conditions.
- Original manuscript targets remain unchanged and are comparison values only.
- Initial completion stage: 50 ordered holdout matches per E06-E08 condition.
  All condition outputs are resumable and may later be extended toward N=200.
- Compute policy: use physical GPUs 0 and 1 only. Evaluators apply the 84%
  per-process CUDA allocator cap. Never run two evaluators on one GPU.

## Immutable completed evidence

- E05 random: `e05_controlled_200/random/results.json` (200/200)
- E05 classic Markov: `e05_controlled_200/classic_markov/results.json` (200/200)
- E05 plain MCTS: `e05_capped_200/plain_mcts/results.json` (200/200)
- E05 priors MCTS: `e05_capped_200/priors_mcts/results.json` (200/200)
- E09 plain-MCTS context analysis: `E09_plain_mcts/results.json` (derived from
  the completed E05 plain-MCTS rows)

## Active/queued adapted sweeps

- E06: `E06_plain_mcts_n50/`
  - `step_minutes_01/`
  - `step_minutes_05/`
  - `step_minutes_10/`
- E07: `E07_plain_mcts_n50/`
  - `num_simulations_50/` runs first, followed by 30, 10, 5, and 1.
- E08: `E08_plain_mcts_n50/`
  - `lookahead_tokens_000/`
  - `lookahead_tokens_040/`
  - `lookahead_tokens_200/`
  - `lookahead_tokens_400/`

Each active condition stores per-match evidence in `results.json` and process
output in `stdout.log`. The evaluator writes after every match and uses
`--resume`; partial files are evidence and must not be deleted.

## Final organization checklist

1. Verify every intended condition has `complete: true` and the declared N.
2. Preserve commands/configuration, stdout, timing, GPU-memory logs, and hashes.
3. Build one E05 four-policy comparison table.
4. Build E06, E07, and E08 sweep tables and plots from measured JSON only.
5. Include E09 possession-bin counts and means.
6. Compare measured values with original claims without copying targets into
   measured-result fields.
7. State the N=50 staging and same-holdout planner-selection caveat prominently.
