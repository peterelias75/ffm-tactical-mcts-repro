# E10 — active-intervention recovery window

This completed experiment asks whether the plain-MCTS coach should be allowed
to issue another active intervention immediately or should wait for the last
action to take effect. State evaluation still occurs every minute. When the
selected action is not `ACTION::NOOP`, subsequent decision points are forced to
no action for 1, 5, or 10 minutes.

## Fixed configuration

- 50 matched held-out second halves
- possession-only objective
- plain MCTS with uniform priors
- five simulations per eligible decision
- 200 rollout tokens per candidate
- seed `20260727`
- paired no-action values reused from the completed E08 200-token condition

## Final results

| Cooldown | Mean gain | Improved | Gain when improved | 90% LCB |
|---:|---:|---:|---:|---:|
| 1 min (E08 baseline) | +2.34 pp | 64% | +6.39 pp | +1.08 pp |
| **5 min** | **+3.42 pp** | **70%** | **+7.83 pp** | **+1.80 pp** |
| 10 min | +2.19 pp | 56% | +6.81 pp | +0.96 pp |

The complete five- and ten-minute per-match records are distributed in their
two subdirectories as lossless `results.json.gz` archives. Run
`gzip -dk results.json.gz` to recover the audited JSON. The one-minute baseline
remains in `../E08_plain_mcts_step1_sim5_n50/lookahead_tokens_200/results.json`.
