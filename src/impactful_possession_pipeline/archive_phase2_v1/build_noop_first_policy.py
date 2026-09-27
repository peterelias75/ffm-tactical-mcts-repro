import argparse
import json
from pathlib import Path

from .utils import load_json, save_json


def action_name(path: Path, data: dict) -> str:
    return data.get("action") or path.stem


def candidate_key(row: dict) -> tuple[str, int]:
    return (row["match"], int(row["cut_minute"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative no-op-first intervention policy table.")
    parser.add_argument("--risk-gates", nargs="*", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-predicted-lcb-pp", type=float, default=0.0)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.625)
    parser.add_argument("--min-risk-gate-score", type=float, default=0.0)
    parser.add_argument("--min-anchor-margin", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    rows_by_state: dict[tuple[str, int], list[dict]] = {}
    gate_summaries = []
    for gate_path in [Path(path) for path in args.risk_gates]:
        data = load_json(gate_path)
        action = action_name(gate_path, data)
        summary = dict(data.get("training_summary", {}))
        summary["path"] = str(gate_path)
        summary["action"] = action
        summary["recommendation"] = data.get("recommendation", {})
        gate_summaries.append(summary)
        for row in data.get("top_candidates", []):
            item = {
                "match": row["match"],
                "match_json": row["match_json"],
                "cut_minute": int(row["cut_minute"]),
                "end_minute": int(row.get("end_minute", int(row["cut_minute"]) + 5)),
                "action": row["action"],
                "predicted_lcb90_pp": float(row["predicted_lcb90_pp"]),
                "predicted_positive_seed_rate": float(row["predicted_positive_seed_rate"]),
                "predicted_std_delta_pp": float(row["predicted_std_delta_pp"]),
                "risk_gate_score": float(row["risk_gate_score"]),
                "anchor_distance_margin": float(row["anchor_distance_margin"]),
                "robust_anchor_distance": float(row["robust_anchor_distance"]),
                "failed_anchor_distance": float(row["failed_anchor_distance"]),
            }
            rows_by_state.setdefault(candidate_key(item), []).append(item)

    decisions = []
    eval_ready = []
    for key, candidates in rows_by_state.items():
        ranked = sorted(
            candidates,
            key=lambda row: (
                row["predicted_lcb90_pp"],
                row["predicted_positive_seed_rate"],
                row["risk_gate_score"],
            ),
            reverse=True,
        )
        best = ranked[0]
        passes = (
            best["predicted_lcb90_pp"] >= args.min_predicted_lcb_pp
            and best["predicted_positive_seed_rate"] >= args.min_positive_seed_rate
            and best["risk_gate_score"] >= args.min_risk_gate_score
            and best["anchor_distance_margin"] >= args.min_anchor_margin
        )
        decision = {
            "match": key[0],
            "cut_minute": key[1],
            "selected_action": best["action"] if passes else "ACTION::NOOP",
            "status": "candidate_for_lcb_validation" if passes else "noop_by_safety_gate",
            "best_candidate": best,
            "ranked_candidates": ranked,
        }
        decisions.append(decision)
        if passes:
            eval_ready.append(decision)

    decisions = sorted(
        decisions,
        key=lambda row: (
            row["status"] == "candidate_for_lcb_validation",
            row["best_candidate"]["predicted_lcb90_pp"],
            row["best_candidate"]["risk_gate_score"],
        ),
        reverse=True,
    )
    eval_ready = sorted(
        eval_ready,
        key=lambda row: (
            row["best_candidate"]["predicted_lcb90_pp"],
            row["best_candidate"]["risk_gate_score"],
        ),
        reverse=True,
    )
    output = {
        "policy": "noop_first_risk_gate",
        "thresholds": {
            "min_predicted_lcb_pp": args.min_predicted_lcb_pp,
            "min_positive_seed_rate": args.min_positive_seed_rate,
            "min_risk_gate_score": args.min_risk_gate_score,
            "min_anchor_margin": args.min_anchor_margin,
        },
        "gate_summaries": gate_summaries,
        "states_considered": len(decisions),
        "intervention_candidates": len(eval_ready),
        "noop_decisions": len(decisions) - len(eval_ready),
        "recommended_validation": eval_ready[: args.top_k],
        "policy_table": decisions[: args.top_k],
        "decision": (
            "validate_candidates_before_deployment"
            if eval_ready
            else "default_to_noop_until_more_labeled_actions_or_better_risk_model"
        ),
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "states_considered": output["states_considered"],
        "intervention_candidates": output["intervention_candidates"],
        "noop_decisions": output["noop_decisions"],
        "decision": output["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
