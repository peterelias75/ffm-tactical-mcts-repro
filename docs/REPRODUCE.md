# Fast reproduction and validation

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Set these paths for your installation:

```bash
export FFM_CHECKPOINT=/path/to/best_forward_simulator.pt
export STATSBOMB_EVENTS=/path/to/statsbomb/open-data/data/events
export STATSBOMB_LINEUPS=/path/to/statsbomb/open-data/data/lineups
```

## Seconds: static validation

```bash
python -m py_compile src/impactful_possession_pipeline/*.py experiments/*.py
python scripts/summarize_results.py
```

The summary command reads committed JSON only and does not require a GPU.

## Minutes: one-match smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python -m impactful_possession_pipeline.evaluate_mcts_pipeline \
  --checkpoint "$FFM_CHECKPOINT" \
  --events-dir "$STATSBOMB_EVENTS" \
  --lineups-dir "$STATSBOMB_LINEUPS" \
  --holdout-json data/holdouts/holdout_table2_200.json \
  --max-matches 1 --num-simulations 1 --step-minutes 10 \
  --rollout-lookahead-tokens 40 --uniform-prior-weight 1.0 \
  --out-json /tmp/ffm_smoke.json
```

This checks loading, encoding, legal-action generation, rollout, MCTS selection,
metric calculation, and incremental persistence. It is not a paper result.

## Full best configuration

```bash
CUDA_VISIBLE_DEVICES=0 python -m impactful_possession_pipeline.evaluate_mcts_pipeline \
  --checkpoint "$FFM_CHECKPOINT" \
  --events-dir "$STATSBOMB_EVENTS" \
  --lineups-dir "$STATSBOMB_LINEUPS" \
  --holdout-json data/holdouts/holdout_table2_200.json \
  --max-matches 50 --num-simulations 5 --step-minutes 1 \
  --intervention-cooldown-minutes 1 \
  --rollout-lookahead-tokens 200 --uniform-prior-weight 1.0 \
  --selection-policy visits --reward-mode possession \
  --seed 20260727 --resume --out-json results.json
```

For E10, change only `--intervention-cooldown-minutes` to `5` or `10`.

## Expected best-run target

Small differences may occur across PyTorch/CUDA versions, but the committed run
records the target aggregate: +2.34 pp mean paired possession gain, 64% matches
improved, and +1.08 pp 90% bootstrap lower confidence bound.

## Resume and memory safety

- Always pass `--resume`; each completed match is saved immediately.
- Use one process per GPU.
- The evaluator caps its CUDA allocator fraction.
- Do not run two writers against the same output JSON.
