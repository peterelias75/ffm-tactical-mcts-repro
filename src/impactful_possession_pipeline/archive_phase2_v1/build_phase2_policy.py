import argparse
import json
from pathlib import Path

from .utils import load_json, save_json


def load_optional(path: str | None) -> dict:
    if not path:
        return {}
    return load_json(path) if Path(path).exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative Phase 2 receding-horizon intervention policy artifact.")
    parser.add_argument("--decision-report", required=True)
    parser.add_argument("--risk-gates", nargs="*", default=[])
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-selection-seeds", type=int, default=8)
    parser.add_argument("--min-lcb-pp", type=float, default=0.0)
    args = parser.parse_args()

    decision = load_optional(args.decision_report)
    gates = [load_optional(path) for path in args.risk_gates]
    ready_candidates = []
    for gate in gates:
        for row in gate.get("eval_ready_candidates", []):
            if (
                row.get("predicted_lcb90_pp", 0.0) > args.min_lcb_pp
                and row.get("predicted_positive_seed_rate", 0.0) >= 0.625
                and row.get("risk_gate_score", 0.0) > 0.0
            ):
                ready_candidates.append(row)

    ready_candidates = sorted(
        ready_candidates,
        key=lambda row: (row.get("risk_gate_score", 0.0), row.get("predicted_lcb90_pp", 0.0)),
        reverse=True,
    )
    selected = ready_candidates[0] if ready_candidates else None
    policy = {
        "policy_name": "phase2_receding_horizon_lcb_gate",
        "status": "ready_for_candidate_eval" if selected else "hold_noop_only",
        "default_action": "ACTION::NOOP",
        "horizon_minutes": 5,
        "replan_every_minutes": 5,
        "selection_rule": {
            "min_selection_seeds": args.min_selection_seeds,
            "min_lcb_pp": args.min_lcb_pp,
            "min_positive_seed_rate": 0.625,
            "must_pass_risk_gate": True,
            "allow_generalized_intervention": bool(selected),
        },
        "selected_candidate": selected,
        "eval_ready_candidates": ready_candidates,
        "confirmed_evidence": decision.get("confirmed_positive"),
        "current_limitations": decision.get("failed_generalization"),
        "risk_gate_statuses": [
            {
                "action": gate.get("action"),
                "recommendation": gate.get("recommendation"),
                "label_rows": gate.get("label_rows"),
                "robust_label_rows": gate.get("robust_label_rows"),
                "failed_or_noisy_label_rows": gate.get("failed_or_noisy_label_rows"),
            }
            for gate in gates
        ],
        "next_required_data": (
            "Collect more independent robust-positive anchors. Current generalized policy must stay NOOP-only."
            if not selected
            else "Run paired rollout validation for selected_candidate before allowing deployment."
        ),
        "source_artifacts": {
            "decision_report": args.decision_report,
            "risk_gates": args.risk_gates,
        },
    }
    save_json(policy, args.out_json)
    print(json.dumps({
        "status": policy["status"],
        "default_action": policy["default_action"],
        "selected_candidate": policy["selected_candidate"],
        "next_required_data": policy["next_required_data"],
    }, indent=2))


if __name__ == "__main__":
    main()
