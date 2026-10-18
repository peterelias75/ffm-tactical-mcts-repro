#!/usr/bin/env python3
import runpy
from pathlib import Path
from chartlib import build

KEYS = [
    "E01_forward_model", "E02_token_augmentation", "E04_repeat_stability",
    "E05_policy_comparison", "E06_intervention_frequency",
    "E07_search_budget", "E08_lookahead",
]
for key in KEYS:
    build(key)
for directory in (
    "E03_spatial_grid", "Figure6_mcts_sweeps", "Figure7_match_context",
    "Figure8_objectives", "E10_intervention_cooldown",
):
    runpy.run_path(
        str(Path(__file__).resolve().parent / directory / "build_chart.py"),
        run_name="__main__",
    )
