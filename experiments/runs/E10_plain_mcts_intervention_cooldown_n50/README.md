# Plain-MCTS intervention cooldown experiment

Common configuration: 50 held-out matches, possession objective, decisions every
1 minute, 5 simulations per eligible decision, 200-token lookahead, and uniform
priors (plain MCTS).

- Cooldown 1 minute is the already validated baseline at
  `paper_reproduction/runs/E08_plain_mcts_step1_sim5_n50/lookahead_tokens_200/results.json`.
- Cooldown 5 minutes writes to `cooldown_5/results.json`.
- Cooldown 10 minutes writes to `cooldown_10/results.json`.

After an active intervention at minute `t`, decisions are still logged each
minute, but minutes before `t + cooldown` are forced to `ACTION::NOOP`.
