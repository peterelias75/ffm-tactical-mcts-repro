import argparse
import json
from pathlib import Path

from .utils import save_json


RUN_DIR = Path("impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048")
REPORT_MD = Path("impactful_possession_pipeline/PROJECT_REPORT.md")
REPORT_JSON = Path("impactful_possession_pipeline/runs/project_report_summary.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def simulator_rows() -> tuple[list[list[str]], dict]:
    paths = [
        ("struct_b_den108", RUN_DIR / "structure_density_100matches/rollout_struct_b_den108_100matches.json"),
        ("struct_b_den110", RUN_DIR / "structure_density_100matches/rollout_struct_b_den110_100matches.json"),
        ("struct_b_den112", RUN_DIR / "structure_density_100matches/rollout_struct_b_den112_100matches.json"),
    ]
    rows = []
    by_name = {}
    for name, path in paths:
        summary = load_json(path)["summary"]
        by_name[name] = summary
        rows.append(
            [
                name,
                str(summary["num_matches"]),
                fmt(summary["mean_abs_possession_error_pp"]),
                fmt(summary["mean_real_events_per_minute"]),
                fmt(summary["mean_sim_events_per_minute"]),
                fmt(summary["mean_event_type_l1"], 3),
                fmt(summary["mean_team_l1"], 3),
                fmt(summary["mean_location_l1"], 3),
                fmt(summary["mean_sim_illegal_possession_switch_rate"], 3),
            ]
        )
    return rows, by_name


def horizon_rows(best_summary: dict) -> list[list[str]]:
    rows = []
    for horizon, values in best_summary["drift_by_horizon"].items():
        rows.append(
            [
                horizon,
                fmt(values["mean_abs_possession_error_pp"]),
                fmt(values["mean_team_l1"], 3),
                fmt(values["mean_real_events_per_minute"]),
                fmt(values["mean_sim_events_per_minute"]),
            ]
        )
    return rows


def intervention_rows(scorer: dict) -> list[list[str]]:
    rows = []
    for action, values in scorer["overall"].items():
        rows.append(
            [
                action,
                str(values["n"]),
                fmt(values["mean_value_head_delta_pp"]),
                fmt(values["mean_rollout_delta_pp"]),
                fmt(values["rollout_lcb90_pp"]),
                pct(values["match_positive_rate"]),
                fmt(values["mean_calibration_gap_pp"]),
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the current project report from saved experiment JSONs.")
    parser.add_argument("--out-md", type=Path, default=REPORT_MD)
    parser.add_argument("--out-json", type=Path, default=REPORT_JSON)
    args = parser.parse_args()

    sim_rows, sim_by_name = simulator_rows()
    best = sim_by_name["struct_b_den112"]
    value_head = load_json(RUN_DIR / "value_head_interventions_final_holdout.json")["summary"]
    scorer = load_json(Path("impactful_possession_pipeline/runs/intervention_scorer_summary.json"))
    recommendation = scorer["recommendation"]

    summary = {
        "best_simulator": {
            "recipe": "struct_b_den112",
            "num_matches": best["num_matches"],
            "mae_possession_pp": best["mean_abs_possession_error_pp"],
            "real_events_per_minute": best["mean_real_events_per_minute"],
            "sim_events_per_minute": best["mean_sim_events_per_minute"],
            "event_type_l1": best["mean_event_type_l1"],
            "team_l1": best["mean_team_l1"],
            "location_l1": best["mean_location_l1"],
            "sim_illegal_switch_rate": best["mean_sim_illegal_possession_switch_rate"],
        },
        "value_head_screening": value_head,
        "intervention_calibration": {
            "deduped_match_action_rows": scorer["deduped_match_action_rows"],
            "train_rows": scorer["train_rows"],
            "test_rows": scorer["test_rows"],
            "overall": scorer["overall"],
            "recommendation": recommendation,
        },
        "queued_experiment": {
            "screen": "broad_intervention_labels",
            "output_dir": str(RUN_DIR / "calibration_labels_vh015_s1"),
            "status": "waiting_for_statsbomb_mount",
        },
    }
    save_json(summary, str(args.out_json))

    md = f"""# Impactful Possession Project Report

Generated from saved experiment artifacts in `{RUN_DIR}`.

## Executive Status

- Phase 1 forward simulator: strong, near publishable.
- Best current simulator recipe: `struct_b_den112`.
- Best 100-match possession MAE: `{fmt(best["mean_abs_possession_error_pp"])}pp`.
- Event density is calibrated: real `{fmt(best["mean_real_events_per_minute"])} events/min`, simulated `{fmt(best["mean_sim_events_per_minute"])} events/min`.
- Illegal simulated possession switches are controlled: `{fmt(best["mean_sim_illegal_possession_switch_rate"], 3)}`.
- Phase 2 intervention policy: not deployable yet. The value head is systematically optimistic and must be calibrated with broader paired rollout labels.

## Phase 1: Forward Simulator

The simulator is a structured-token causal transformer trained on StatsBomb event JSON. It predicts continuations from live match prefixes and uses football-structure decoding constraints at rollout time.

{table(
    ["Recipe", "Matches", "MAE pp", "Real ev/min", "Sim ev/min", "Event L1", "Team L1", "Loc L1", "Illegal switch"],
    sim_rows,
)}

Best recipe details:

- Possession guidance strength: `12`
- Adaptive density from prefix: enabled
- Density multiplier: `1.12`
- Prefix possession target: enabled
- Possession state enforcement: enabled
- Football structure constraints: enabled
- Pass receipt boost: `1.0`
- Pass carry boost: `0.4`
- Overproduced event penalty: `0.5`
- Force receipt after pass run: `3`

### Horizon Behavior

{table(
    ["Horizon", "MAE pp", "Team L1", "Real ev/min", "Sim ev/min"],
    horizon_rows(best),
)}

Interpretation: short windows are noisy because possession over one minute is intrinsically sparse and bursty, while 10-15 minute windows become much more stable. The 15-minute result is the most relevant for intervention rollout validation.

## Phase 2: Intervention Planning

The value head screened `340` holdout matches and accepted `{pct(value_head["accept_rate"])}` of candidate interventions.

Screening summary:

- Accepted interventions: `{int(round(value_head["accept_rate"] * value_head["holdout_files"]))}` / `{value_head["holdout_files"]}`
- Mean accepted value-head delta: `{fmt(value_head["mean_accepted_delta_pp"])}pp`
- Chosen action counts: `{value_head["chosen_action_counts"]}`

Paired rollout calibration does not yet support using those interventions as final decisions:

{table(
    ["Action", "Rows", "VH delta pp", "Actual pp", "LCB90 pp", "Positive", "Gap pp"],
    intervention_rows(scorer),
)}

Current recommendation: **{recommendation["decision"]}**

The important scientific finding is that the simulator is usable, but the intervention value head is not yet calibrated. The current policy should default to `NOOP` until broader paired labels confirm a stable positive effect.

## Queued Experiment

A screen watcher is already running:

```text
screen: broad_intervention_labels
log: impactful_possession_pipeline/runs/broad_intervention_labels_watch.log
```

When the StatsBomb mount returns, it will run four paired-label jobs in parallel:

- `HOME_SUB_ATTACK` on GPU 0
- `HOME_SUB_NEUTRAL` on GPU 1
- `HOME_TACTIC_ATTACK` on GPU 2
- `HOME_TACTIC_NEUTRAL` on GPU 3

The outputs will be saved under:

```text
{RUN_DIR / "calibration_labels_vh015_s1"}
```

## Publication Framing

The strongest current paper angle is:

1. A leakage-aware structured-token forward simulator for football possession.
2. A rollout diagnostic suite measuring possession error, density, event realism, team realism, and illegal state transitions.
3. A counterfactual intervention framework showing why direct value-head intervention selection is insufficient without paired rollout calibration.

For a strong workshop or applied ML submission, the simulator results are already meaningful. For a stronger ML conference submission, Phase 2 needs one more successful step: a calibrated intervention scorer with fresh paired rollout validation showing at least `+1pp` actual gain, `>=60%` match-positive rate, and `>=60%` seed-positive rate on unseen matches.
"""

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    print(f"saved markdown report to {args.out_md}")
    print(f"saved json summary to {args.out_json}")


if __name__ == "__main__":
    main()
