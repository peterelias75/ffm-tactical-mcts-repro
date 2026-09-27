set -euo pipefail

CUDA_VISIBLE_DEVICES=0 /home/peter/miniconda3/envs/fb-env/bin/python -u -m impactful_possession_pipeline.evaluate_mcts_pipeline \
  --checkpoint impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt \
  --holdout-json paper_reproduction/holdout_table2_200.json \
  --noop-cache-json paper_reproduction/runs/e05_controlled_200/random/results.json \
  --max-matches 200 --cut-minute 45 --step-minutes 5 --num-simulations 10 \
  --uniform-prior-weight 1.0 --selection-policy visits --risk-penalty 0.0 \
  --noop-margin 0.0 --reward-mode possession --seed 20260727 --resume \
  --out-json paper_reproduction/runs/e05_controlled_200/plain_mcts/results.json

/home/peter/miniconda3/envs/fb-env/bin/python -u -m paper_reproduction.sweep_e07_search_budget \
  --gpu 0 --holdout-json paper_reproduction/holdout_table2_200.json --max-matches 200 \
  --noop-cache-json paper_reproduction/runs/e05_controlled_200/random/results.json \
  --run-root paper_reproduction/runs/e07_search_budget
