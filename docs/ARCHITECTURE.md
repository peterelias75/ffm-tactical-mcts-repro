# Architecture

## System flow

```mermaid
flowchart LR
    A[StatsBomb events + lineups] --> B[Structured encoder]
    B --> C[Event/action token sequence]
    C --> D[Transformer forward model]
    D --> E[Autoregressive match rollout]
    C --> F[Legal tactical actions]
    F --> G[Plain MCTS root search]
    D --> G
    G --> H[Selected intervention]
    H --> C
    E --> I[Possession / shot metrics]
```

The encoder converts real events into structured frames and tokens. At every
eligible decision minute, root-level MCTS allocates a fixed simulation budget
across legal substitutions, tactical formation changes, and no-action. Each
simulation is a free-running model rollout; the chosen action is injected and
the match advances before the next decision.

## Plain-MCTS decision

```mermaid
flowchart TD
    S[Current generated match state] --> L[Enumerate legal actions]
    L --> P[Assign uniform priors]
    P --> U[PUCT selection]
    U --> R[Generate up to N rollout tokens]
    R --> V[Score objective]
    V --> U
    U -->|simulation budget exhausted| C[Choose most visited action]
    C --> A[Advance simulated match]
```

Plain MCTS uses `uniform_prior_weight=1.0`. Priors-guided runs are retained for
comparison, but the final adapted experiments use uniform priors because they
were more reliable with the available checkpoint.

## Cooldown extension

```mermaid
stateDiagram-v2
    [*] --> Eligible
    Eligible --> Eligible: NOOP
    Eligible --> Blocked: active action at minute t
    Blocked --> Blocked: forced NOOP before t + cooldown
    Blocked --> Eligible: minute >= t + cooldown
```

Decision time remains one minute. Cooldown affects only eligibility for another
active action; blocked minutes are still generated and logged.
