# E05 controlled Table 2 reproduction

All four policies use the exact ordered 200-match subset in
`paper_reproduction/holdout_table2_200.json`, itself copied from the modern
checkpoint's persisted test split. No split is recomputed.

Physical GPU allocation:

- GPU 0: random baseline, then plain MCTS
- GPU 1: classic Markov baseline, then priors-guided MCTS

Only one CUDA process runs on each GPU. Every evaluator caps the PyTorch
allocator at 84%, and independent 30-second monitors flag memory above
9,500 MiB. Outputs are persisted after every completed match and use `--resume`.

The classic Markov transition table was fitted on 3,060 non-holdout matches:
`paper_reproduction/runs/E05_classic_markov_table_20260830/results.json`.

See each policy subdirectory's `command.sh`, `stdout.log`, `gpu_memory.csv`,
and eventually `results.json`.

The detached handoff sessions use `paper_reproduction/wait_for_complete_then_run.sh`.
They start an MCTS command only after the corresponding baseline JSON explicitly
contains `"complete": true`; a missing process or incomplete file never advances
the queue.

After plain MCTS, GPU 0 continues into E07. After priors MCTS, GPU 1 derives
E09, validates/copies the identical default-priors result into E06-step-5 and
E08-lookahead-200, then continues through E06 and E08. E07's identical
10-simulation point is intentionally not copied because GPU 0 can reach that
directory concurrently; avoiding two writers takes precedence over reuse.
The reuse helper checks the completed 200-match config and verifies SHA-256
identity for every copy before any sweep treats it as complete.

All later MCTS/sweep conditions also pass a completed baseline JSON through
`--noop-cache-json`. The evaluator verifies completeness, checkpoint identity,
and coverage of all 200 requested match IDs before reuse. Only the deterministic
no-intervention reference is reused; every intervention/search rollout remains
independently computed. This removes one redundant full-half rollout per match.
