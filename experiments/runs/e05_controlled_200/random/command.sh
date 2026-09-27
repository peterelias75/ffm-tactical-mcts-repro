CUDA_VISIBLE_DEVICES=0 /home/peter/miniconda3/envs/fb-env/bin/python -u -m impactful_possession_pipeline.evaluate_baseline_policy_pipeline \
  --policy random \
  --checkpoint impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt \
  --holdout-json paper_reproduction/holdout_table2_200.json \
  --max-matches 200 --cut-minute 45 --step-minutes 5 --seed 20260727 --resume \
  --out-json paper_reproduction/runs/e05_controlled_200/random/results.json
