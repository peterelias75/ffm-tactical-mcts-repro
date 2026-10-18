# Experiment catalog

The exact machine-readable records live under `experiments/runs/`. Result JSON
files contain configuration, per-match rows, aggregate metrics, and completion
flags.

| ID | Question | Status / key result |
|---|---|---|
| E01 | Forward-model architecture comparison | Historical reproduction retained |
| E02 | Token augmentation | Historical reproduction retained |
| E03 | Spatial grid sensitivity | Historical reproduction retained |
| E04 | Repeat stability | Historical reproduction retained |
| E05 | Random, Markov, plain MCTS, priors MCTS | Controlled 200-match comparison complete |
| E06 | Intervention interval: 1/5/10 min | 1 minute best: +2.57 pp on 50 matches |
| E07 | Search budget: 1/5/10/30/50 sims | 5 sims best: +2.87 pp on 50 matches |
| E08 | Rollout tokens: 0/40/200/400 | 200 tokens best: +2.34 pp, 64% improved |
| Fig. 7 | Match context and intervention mix | 50/50 complete; exact deterministic replay |
| Fig. 8 | Shots-only vs balanced objective | 50/50 per condition complete |
| E10 | Active-action cooldown: 1/5/10 min | Complete; 5 min best: +3.42 pp, 70% improved |

## Adapted plain-MCTS sweep

### E06: intervention frequency

| Decision interval | Mean gain | Improved |
|---:|---:|---:|
| 1 min | **+2.57 pp** | 62% |
| 5 min | +1.63 pp | 50% |
| 10 min | +0.85 pp | 60% |

### E07: simulation budget at one-minute decisions

| Simulations | Mean gain | Improved |
|---:|---:|---:|
| 1 | 0.00 pp | 0% |
| 5 | **+2.87 pp** | **70%** |
| 10 | +2.57 pp | 62% |
| 30 | +1.64 pp | 54% |
| 50 | +2.55 pp | 70% |

### E08: rollout-token budget at one minute / five simulations

| Tokens | Mean gain | Improved | 90% bootstrap LCB |
|---:|---:|---:|---:|
| 0 | +1.73 pp | 58% | +0.42 pp |
| 40 | +2.07 pp | 60% | +0.61 pp |
| 200 | **+2.34 pp** | **64%** | **+1.08 pp** |
| 400 | +2.17 pp | 54% | +0.83 pp |

## Figure 7: context

| Real second-half possession | Matches | Coach minus real possession |
|---|---:|---:|
| <45% | 19 | +18.10 pp |
| 45–55% | 13 | +6.81 pp |
| >55% | 18 | −5.34 pp |

The very small number of decisions observed in low simulated-possession states
should be stated when interpreting the intervention-mix panel.

## Figure 8: objective trade-off

| Objective | Shot gain vs no-action | Shot improved | Possession gain vs no-action | Possession improved |
|---|---:|---:|---:|---:|
| Shots-only | +0.38 | 48% | +3.98 pp | 78% |
| Balanced | +0.18 | 40% | +2.94 pp | 66% |

The simulator underproduces shots (0.62 no-action simulated home shots versus
5.46 real), so shot conclusions must use the paired simulated baseline and be
reported as a limitation.


## E10: recovery time after an active intervention

E10 keeps the planner eligible to inspect the state every minute, but after a
non-no-op action it forces no-action decisions until the configured recovery
window expires. This tests whether tactical actions benefit from time to take
effect before the coach changes the plan again. All conditions use the same 50
matches, seed, paired no-action values, plain uniform-prior MCTS, five
simulations per eligible decision, and 200-token candidate rollouts.

| Active-action cooldown | Mean gain | Improved | Gain when improved | 90% bootstrap LCB |
|---:|---:|---:|---:|---:|
| 1 min | +2.34 pp | 64% | +6.39 pp | +1.08 pp |
| **5 min** | **+3.42 pp** | **70%** | **+7.83 pp** | **+1.80 pp** |
| 10 min | +2.19 pp | 56% | +6.81 pp | +0.96 pp |

The five-minute recovery window performs best on every primary aggregate. The
ten-minute window is too restrictive: it reduces both mean gain and the share
of improved matches below the one-minute baseline. These are paired simulated
policy effects and should not be interpreted as causal estimates from real
coaching interventions.
