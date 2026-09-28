# Reproduction and validation strategy

Validation has three levels because checking code and saved evidence should not require repeating every multi-hour GPU sweep: inspect committed artifacts, test the full path on one match, and only then launch a complete condition.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Provide local data and checkpoint paths:

```bash
export FFM_CHECKPOINT=/path/to/best_forward_simulator.pt
export STATSBOMB_EVENTS=/path/to/statsbomb/open-data/data/events
export STATSBOMB_LINEUPS=/path/to/statsbomb/open-data/data/lineups
```

## Level 1 — seconds: artifact validation

```bash
python -m py_compile src/impactful_possession_pipeline/*.py experiments/*.py
python scripts/summarize_results.py
```

This verifies that source files compile and that the committed best-result JSON is present, parseable, complete, and equal to the headline values. It does not test model generation.

## Level 2 — minutes: one-match smoke test

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

This checks checkpoint loading, encoding, legal actions, generation, MCTS selection, metrics, and persistence. It is intentionally cheap and is not a paper result.

## Level 3 — full best configuration

This recreates the configuration selected sequentially by E06, E07, and E08. Holding non-target variables fixed matters: changing decision interval, simulations, and rollout length together would make the source of a gain unidentifiable.

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

## Reading a result

Check `summary.complete` before accepting aggregates. `mean_delta_pp` is MCTS possession minus paired no-action possession, multiplied by 100. Improvement rate counts strictly positive match deltas. The bootstrap lower confidence bound is a conservative uncertainty check, not external validation.

The committed best-run target is +2.34 pp mean paired gain, 64% improved, and +1.08 pp 90% bootstrap LCB. Small numerical differences may occur across PyTorch/CUDA versions.

## Resume and memory safety

- Pass `--resume`; each completed match is saved immediately.
- Use one process per GPU.
- The evaluator caps its CUDA allocator fraction.
- Never run two writers against one output JSON.
- Preserve the holdout order and seed when resuming.
- Record incomplete runs separately from complete conditions.

These are methodological safeguards, not only operational advice: fixed seeds and resume-safe outputs prevent interruptions from silently changing the stochastic sample.
