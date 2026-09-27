set -euo pipefail

CUDA_VISIBLE_DEVICES=1 /home/peter/miniconda3/envs/fb-env/bin/python -u -m impactful_possession_pipeline.evaluate_mcts_pipeline \
  --checkpoint impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt \
  --holdout-json paper_reproduction/holdout_table2_200.json \
  --noop-cache-json paper_reproduction/runs/e05_controlled_200/classic_markov/results.json \
  --max-matches 200 --cut-minute 45 --step-minutes 5 --num-simulations 10 \
  --uniform-prior-weight 0.05 --selection-policy visits --risk-penalty 0.0 \
  --noop-margin 0.0 --reward-mode possession --seed 20260727 --resume \
  --out-json paper_reproduction/runs/e05_controlled_200/priors_mcts/results.json

/home/peter/miniconda3/envs/fb-env/bin/python -u -m paper_reproduction.aggregate_e09_match_context \
  --results-json paper_reproduction/runs/e05_controlled_200/priors_mcts/results.json \
  --out-json paper_reproduction/runs/e09_match_context/results.json

/home/peter/miniconda3/envs/fb-env/bin/python -u -m paper_reproduction.reuse_default_priors_condition \
  --source paper_reproduction/runs/e05_controlled_200/priors_mcts/results.json \
  --target paper_reproduction/runs/e06_intervention_frequency/step_minutes_05/results.json \
  --target paper_reproduction/runs/e08_lookahead/lookahead_tokens_200/results.json \
  --manifest paper_reproduction/runs/default_priors_condition_reuse.json

/home/peter/miniconda3/envs/fb-env/bin/python -u -m paper_reproduction.sweep_e06_intervention_frequency \
  --gpu 1 --holdout-json paper_reproduction/holdout_table2_200.json --max-matches 200 \
  --noop-cache-json paper_reproduction/runs/e05_controlled_200/classic_markov/results.json \
  --run-root paper_reproduction/runs/e06_intervention_frequency

/home/peter/miniconda3/envs/fb-env/bin/python -u -m paper_reproduction.sweep_e08_lookahead \
  --gpu 1 --holdout-json paper_reproduction/holdout_table2_200.json --max-matches 200 \
  --noop-cache-json paper_reproduction/runs/e05_controlled_200/classic_markov/results.json \
  --run-root paper_reproduction/runs/e08_lookahead
