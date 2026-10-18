# E10 cooldown figures

Run `python build_chart.py` to regenerate two opaque-white PNG bar charts. The
script stops unless all cooldown conditions are complete with `n=50`.

- `cooldown_performance_n50.png`: improvement-rate bars with average possession
  gain and gain-when-improved printed inside each bar.
- `cooldown_intervention_patterns_n50.png`: grouped bars across match-minute
  windows, with bar colors identifying the 1-, 5-, and 10-minute cooldowns.

No-action decisions are excluded because this chart shows when active
interventions were issued.
