# E01 full leakage-safe Table 1 evaluation

Checkpoint: `impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt`

The checkpoint contains an exact persisted 340-match test split. GPU 0 evaluates
absolute indices 0-169 and GPU 1 evaluates 170-339. Both commands use seed 42;
the runner derives each match seed as `42 + absolute_test_index`, so sharding and
resume do not change generated samples.

```bash
CUDA_VISIBLE_DEVICES=0 /home/peter/miniconda3/envs/fb-env/bin/python -u -m paper_reproduction.run_forward_shard \
  --checkpoint impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt \
  --offset 0 --count 170 --seed 42 --resume \
  --out-json paper_reproduction/runs/table1_full340_20260830/gpu0_offset000/results.json

CUDA_VISIBLE_DEVICES=1 /home/peter/miniconda3/envs/fb-env/bin/python -u -m paper_reproduction.run_forward_shard \
  --checkpoint impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt \
  --offset 170 --count 170 --seed 42 --resume \
  --out-json paper_reproduction/runs/table1_full340_20260830/gpu1_offset170/results.json

bash paper_reproduction/wait_and_aggregate_forward.sh \
  paper_reproduction/runs/table1_full340_20260830/combined_results.json \
  paper_reproduction/runs/table1_full340_20260830/gpu0_offset000/results.json \
  paper_reproduction/runs/table1_full340_20260830/gpu1_offset170/results.json
```

Each CUDA process caps its allocator at 84% of its visible 11,264 MiB GPU.
Independent monitors sample every 30 seconds and flag usage above 9,500 MiB.
Only physical GPUs 0 and 1 are used.

## Compatible-row reuse

Twelve rows computed earlier with this same checkpoint, persisted test split,
absolute index seed mapping, decoding configuration, and metric implementation
were copied into the resumable shards with `paper_reproduction.seed_forward_shards`:
11 belong to indices 0-10 and one belongs to index 175. The utility checks that
every reused match filename occurs in the exact target checkpoint slice and that
all five required error metrics are present. Remaining rows are generated normally.

The paper reports N=450; only 340 genuinely held-out matches are available from
this trained checkpoint. The final comparison must retain that sample-size caveat.
