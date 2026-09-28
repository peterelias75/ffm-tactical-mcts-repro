# Experiment catalog and rationale

The experiments form a sequence: validate the simulator, compare policy families, isolate MCTS controls, and then study context, objectives, and cooldowns. Exact per-match records and configurations live under `experiments/runs/`.

## Overview

| ID | Main question | Scope |
|---|---|---|
| E01 | Which forward-model architecture is useful? | Historical architecture comparison |
| E02 | Do richer structured tokens preserve match context? | Token augmentation |
| E03 | What spatial resolution balances detail and support? | Pitch-grid comparison |
| E04 | Are rollout conclusions stable? | Repeated stochastic generation |
| E05 | Does search beat random/Markov policies, and do priors help? | Four policy families |
| E06 | How often should the planner reconsider? | 1/5/10-minute decisions |
| E07 | How much search is useful? | 1/5/10/30/50 simulations |
| E08 | How far should candidate rollouts look? | 0/40/200/400 tokens |
| Figure 7 | Does behavior adapt to match context? | Possession strata and action mix |
| Figure 8 | Can objective weights steer behavior? | Shots-only vs balanced |
| E10 | Should active interventions have recovery time? | 1/5/10-minute cooldowns |

## E01–E04: establish the simulator

### E01 — architecture

**Question.** Which sequence design produces useful football-event continuations?

**Reason.** Tactical search inherits simulator bias. Architecture must be assessed before policy gains are interpreted.

**Design.** Compare historical forward-model variants on held-out continuation and event statistics. These artifacts are labelled historical and are not silently mixed with later adapted runs.

### E02 — token augmentation

**Question.** Do richer state/action tokens preserve football context?

**Reason.** Event type alone omits team, possession, spatial, formation, and intervention state.

**Design.** Add structured token groups while holding evaluation fixed, then inspect event and possession behavior.

### E03 — spatial grid

**Question.** How much pitch resolution is useful without creating sparse tokens?

**Reason.** Coarse grids erase tactical location; fine grids weaken support per token.

**Design.** Compare coarse, regular, and fine partitions. Heatmaps visualize real-versus-simulated density rather than policy reward.

### E04 — repeat stability

**Question.** Are conclusions stable across stochastic generations?

**Reason.** One autoregressive continuation may be atypical.

**Design.** Repeat controlled-seed rollouts and measure dispersion. This motivates paired baselines in later work.

## E05: compare policy families

**Question.** Does search add value beyond random actions and a fixed Markov heuristic, and do learned action priors help?

**Design.** Random, classic Markov, plain MCTS, and priors-guided MCTS use the same matches and fixed budgets.

**Interpretation.** Plain MCTS was more dependable with this checkpoint, so subsequent adapted sweeps optimize it rather than mixing policy families.

## E06–E08: isolate MCTS controls

### E06 — decision interval

Only the decision interval changes: 1, 5, or 10 minutes. Frequent decisions react quickly but may compound model noise; sparse decisions are cheaper but can miss useful moments.

| Interval | Mean paired gain | Improved |
|---:|---:|---:|
| 1 min | **+2.57 pp** | 62% |
| 5 min | +1.63 pp | 50% |
| 10 min | +0.85 pp | 60% |

### E07 — simulation budget

With one-minute decisions fixed, only simulations per root search change. More stochastic search is not automatically better.

| Simulations | Mean paired gain | Improved |
|---:|---:|---:|
| 1 | 0.00 pp | 0% |
| 5 | **+2.87 pp** | **70%** |
| 10 | +2.57 pp | 62% |
| 30 | +1.64 pp | 54% |
| 50 | +2.55 pp | 70% |

### E08 — rollout-token budget

With one-minute decisions and five simulations fixed, only the generated-token cap changes. Short horizons can miss delayed effects; long horizons can drift from the action.

| Tokens | Mean paired gain | Improved | 90% bootstrap LCB |
|---:|---:|---:|---:|
| 0 | +1.73 pp | 58% | +0.42 pp |
| 40 | +2.07 pp | 60% | +0.61 pp |
| 200 | **+2.34 pp** | **64%** | **+1.08 pp** |
| 400 | +2.17 pp | 54% | +0.83 pp |

## Figure 7: behavior by context

**Question.** Does the policy take more risk in difficult states and protect favorable ones?

**Design.** Panel (a) groups matches by real second-half home possession. Panel (b) groups individual decisions by simulated possession immediately before choosing. Saved decisions are replayed deterministically, avoiding a second MCTS sample.

**Reason.** Aggregate gain cannot reveal whether behavior is context-sensitive.

| Real possession | Matches | Coach minus real possession |
|---|---:|---:|
| <45% | 19 | +18.10 pp |
| 45–55% | 13 | +6.81 pp |
| >55% | 18 | −5.34 pp |

Low simulated-possession decisions are few, so the action-mix panel must be interpreted cautiously.

## Figure 8: objective trade-off

**Question.** Can reward weights steer the planner toward shots or ball control?

**Design.** On 50 real low-possession matches, compare shots-only with equal shot/possession weights. Search components use comparable 0–1 shares; outcomes report raw shot-count and possession differences against paired no-action simulations.

| Objective | Shot gain | Shot improved | Possession gain | Possession improved |
|---|---:|---:|---:|---:|
| Shots-only | +0.38 | 48% | +3.98 pp | 78% |
| Balanced | +0.18 | 40% | +2.94 pp | 66% |

The simulator underproduces shots (0.62 no-action simulated home shots versus 5.46 real), so conclusions use the paired simulated baseline and explicitly retain this limitation.

## E10: recovery after an active intervention

**Question.** Does preventing rapid consecutive active interventions improve possession?

**Design.** The planner observes and advances every minute. After a non-no-action choice at minute `t`, subsequent decisions are forced to no-action until `t+1`, `t+5`, or `t+10`. Plain MCTS, five simulations, 200 rollout tokens, matches, and seeds remain fixed.

**Reason.** E06 changes how often any decision occurs. E10 asks a different operational question: how long to wait after actually changing the team. It separates continuous monitoring from intervention recovery.
