import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DataConfig
from .evaluate_action_target_blend import cvar_bottom, sem_lcb
from .select_tail_risk_candidates import checkpoint_test_files, make_matrix, parse_minutes
from .train_context_intervention_scorer import ACTION_ORDER, checkpoint_event_map, context_features
from .utils import load_json, mean, save_json, std


DEFAULT_CHECKPOINT = (
    "impactful_possession_pipeline/runs/"
    "rich_actions_3400matches_finetune_3epochs_20260712_maxlen2048/"
    "best_forward_simulator.pt"
)
DEFAULT_POSITIVE_JSON = "impactful_possession_pipeline/runs/phase2_action_target_blend_3900498_cmid_16seed_summary_20260716.json"
DEFAULT_NEGATIVE_JSON = "impactful_possession_pipeline/runs/phase2_action_target_blend_batch5_cmid_4seeds_20260716.json"
DEFAULT_ANCHOR_JSONS = [
    "impactful_possession_pipeline/runs/phase2_action_blend_compatible_1_3879606_55_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_compatible_2_3879657_45_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_compatible_1_3879606_55_extra4_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_compatible_v2_1_3900498_40_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_guarded_v2_1_3900555_45_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_guarded_strict_v2_1_3890370_45_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_posdelta_strict_1_3829436_45_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_posdelta_strict_2_68322_55_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_posdelta_strict_3_3857281_45_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_posdelta_strict_3_3857281_45_extra4_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_lowtemp_3857281_45_4seeds_20260716.json",
    "impactful_possession_pipeline/runs/phase2_action_blend_v3_strict_1_3825670_40_4seeds_20260717.json",
]


def action_name(action: str) -> str:
    return action.replace("ACTION::", "")


def summarize_deltas(deltas: Sequence[float]) -> dict:
    values = [float(value) for value in deltas]
    return {
        "n": len(values),
        "mean_delta_pp": mean(values),
        "std_delta_pp": std(values),
        "lcb90_delta_pp": sem_lcb(values),
        "cvar25_delta_pp": cvar_bottom(values),
        "worst_delta_pp": min(values) if values else 0.0,
        "positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in values]),
    }


def load_positive_anchor(path: str, event_by_name: dict[str, str]) -> dict:
    data = load_json(path)
    if "match_json" not in data and data.get("sources"):
        source_rows = [load_json(source) for source in data["sources"]]
        data = dict(source_rows[0])
        data["rows"] = [row for source in source_rows for row in source.get("rows", [])]
    match_json = event_by_name.get(Path(data["match_json"]).name, data["match_json"])
    deltas = data.get("deltas_pp") or [row["delta_pp"] for row in data["rows"]]
    return {
        "kind": "positive_anchor",
        "source": path,
        "match": Path(match_json).name,
        "match_json": match_json,
        "action": data["action"],
        "cut_minute": int(data["cut_minute"]),
        "end_minute": int(data["end_minute"]),
        "step_minutes": int(data["step_minutes"]),
        **summarize_deltas(deltas),
    }


def load_negative_anchors(path: str, event_by_name: dict[str, str], max_anchors: int) -> list[dict]:
    data = load_json(path)
    anchors = []
    active_rows = data.get("active_rows") or []
    if active_rows and data.get("active_match"):
        match = f"{data['active_match']}.json"
        match_json = event_by_name.get(match)
        anchors.append(
            {
                "kind": "hard_negative",
                "source": path,
                "match": match,
                "match_json": match_json,
                "action": data.get("action", "ACTION::HOME_SUB_ROLE_CENTRAL_MIDFIELDER"),
                "cut_minute": int(data.get("cut_minute", 45)),
                "end_minute": int(data.get("end_minute", 50)),
                "step_minutes": int(data.get("step_minutes", 5)),
                **summarize_deltas([row["delta_pp"] for row in active_rows]),
            }
        )
    for result in data.get("results", []):
        match_json = event_by_name.get(Path(result["match_json"]).name, result["match_json"])
        anchors.append(
            {
                "kind": "negative_result",
                "source": path,
                "match": Path(match_json).name,
                "match_json": match_json,
                "action": data.get("action", "ACTION::HOME_SUB_ROLE_CENTRAL_MIDFIELDER"),
                "cut_minute": int(data.get("cut_minute", 45)),
                "end_minute": int(data.get("end_minute", 50)),
                "step_minutes": int(data.get("step_minutes", 5)),
                **summarize_deltas([row["delta_pp"] for row in result["rows"]]),
            }
        )
    return [row for row in anchors if row.get("match_json")][:max_anchors]


def load_eval_anchor(path: str, event_by_name: dict[str, str]) -> dict | None:
    if not Path(path).exists():
        return None
    data = load_json(path)
    if "rows" not in data or not data["rows"] or "match_json" not in data:
        return None
    deltas = [row["delta_pp"] for row in data["rows"]]
    summary = summarize_deltas(deltas)
    kind = "positive_eval_anchor" if summary["worst_delta_pp"] >= 0.0 and summary["cvar25_delta_pp"] >= 0.0 else "negative_eval_anchor"
    match_json = event_by_name.get(Path(data["match_json"]).name, data["match_json"])
    return {
        "kind": kind,
        "source": path,
        "match": Path(match_json).name,
        "match_json": match_json,
        "action": data["action"],
        "cut_minute": int(data["cut_minute"]),
        "end_minute": int(data["end_minute"]),
        "step_minutes": int(data["step_minutes"]),
        **summary,
    }


def row_features(anchor: dict) -> dict:
    events = load_json(anchor["match_json"])
    row = {
        "value_head_delta_pp": 0.0,
        "mean_noop_home_possession": 0.5,
        "action": action_name(anchor["action"]),
    }
    return context_features(events, row["action"], row, anchor["cut_minute"])


def feature_names(rows: Sequence[dict]) -> list[str]:
    names = set()
    for row in rows:
        names.update(row["features"])
    return ["bias"] + sorted(name for name in names if name != "bias")


def standardize_from_reference(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    mu[0] = 0.0
    sigma[0] = 1.0
    return (x - mu) / sigma, mu, sigma


def distance(vector: np.ndarray, examples: Sequence[np.ndarray]) -> float:
    if not examples:
        return 0.0
    return min(float(np.linalg.norm(vector - example)) for example in examples)


def make_candidate_rows(
    checkpoint: str,
    event_by_name: dict[str, str],
    action: str,
    minutes: Sequence[int],
    max_matches: int,
    horizon_minutes: int,
) -> list[dict]:
    candidates = []
    base_row = {
        "value_head_delta_pp": 0.0,
        "mean_noop_home_possession": 0.5,
        "action": action_name(action),
    }
    for match_json in checkpoint_test_files(checkpoint)[:max_matches]:
        event_file = event_by_name.get(Path(match_json).name, match_json)
        if not Path(event_file).exists():
            continue
        events = load_json(event_file)
        for minute in minutes:
            candidates.append(
                {
                    "match": Path(event_file).name,
                    "match_json": event_file,
                    "action": action,
                    "cut_minute": int(minute),
                    "end_minute": int(minute) + horizon_minutes,
                    "step_minutes": horizon_minutes,
                    "features": context_features(events, base_row["action"], base_row, int(minute)),
                }
            )
    return candidates


def build_single_eval_command(args: argparse.Namespace, candidate: dict, index: int) -> str:
    out_json = args.eval_out_template.format(index=index, match=Path(candidate["match"]).stem, minute=candidate["cut_minute"])
    log_path = out_json.replace(".json", ".log")
    return (
        f"CUDA_VISIBLE_DEVICES={args.cuda_device} "
        f"/home/peter/miniconda3/envs/fb-env/bin/python -u -m impactful_possession_pipeline.evaluate_action_target_blend "
        f"--checkpoint {args.checkpoint} "
        f"--match-json {candidate['match_json']} "
        f"--out-json {out_json} "
        f"--action {candidate['action']} "
        f"--cut-minute {candidate['cut_minute']} "
        f"--end-minute {candidate['end_minute']} "
        f"--step-minutes {candidate['step_minutes']} "
        f"--rollout-seeds {args.eval_rollout_seeds} "
        f"--seed {args.eval_seed + index * 10000} "
        f"--target-events-per-minute {args.target_events_per_minute} "
        f"--event-density-tolerance {args.event_density_tolerance} "
        f"--possession-guidance-strength {args.possession_guidance_strength} "
        f"--possession-guidance-prior-events {args.possession_guidance_prior_events} "
        f"--target-blend {args.target_blend} "
        f"--max-target-delta-pp {args.max_target_delta_pp} "
        f"--external-possession-weight {args.external_possession_weight} "
        f"> {log_path} 2>&1"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select states compatible with the action-target blend recipe.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--positive-json", default=DEFAULT_POSITIVE_JSON)
    parser.add_argument("--negative-json", default=DEFAULT_NEGATIVE_JSON)
    parser.add_argument("--anchor-jsons", nargs="*", default=DEFAULT_ANCHOR_JSONS)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--action", default="ACTION::HOME_SUB_ROLE_CENTRAL_MIDFIELDER")
    parser.add_argument("--candidate-minutes", default="40,45,50,55")
    parser.add_argument("--horizon-minutes", type=int, default=5)
    parser.add_argument("--max-candidate-matches", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--eval-top-k", type=int, default=3)
    parser.add_argument("--negative-max-anchors", type=int, default=3)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--eval-rollout-seeds", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=140000)
    parser.add_argument("--eval-out-template", default="impactful_possession_pipeline/runs/phase2_action_blend_compatible_{index}_{match}_{minute}_4seeds_20260716.json")
    parser.add_argument("--target-events-per-minute", type=float, default=22.0)
    parser.add_argument("--event-density-tolerance", type=int, default=2)
    parser.add_argument("--possession-guidance-strength", type=float, default=2.0)
    parser.add_argument("--possession-guidance-prior-events", type=float, default=80.0)
    parser.add_argument("--target-blend", type=float, default=0.5)
    parser.add_argument("--max-target-delta-pp", type=float, default=6.0)
    parser.add_argument("--external-possession-weight", type=float, default=1.0)
    args = parser.parse_args()

    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    positive_anchor = load_positive_anchor(args.positive_json, event_by_name)
    negative_anchors = load_negative_anchors(args.negative_json, event_by_name, args.negative_max_anchors)
    eval_anchors = [row for row in (load_eval_anchor(path, event_by_name) for path in args.anchor_jsons) if row]
    positive_anchors = [positive_anchor] + [row for row in eval_anchors if row["kind"] == "positive_eval_anchor"]
    negative_anchors = negative_anchors + [row for row in eval_anchors if row["kind"] == "negative_eval_anchor"]
    anchors = positive_anchors + negative_anchors
    for anchor in anchors:
        anchor["features"] = row_features(anchor)

    candidates = make_candidate_rows(
        args.checkpoint,
        event_by_name,
        args.action,
        parse_minutes(args.candidate_minutes),
        args.max_candidate_matches,
        args.horizon_minutes,
    )
    excluded = {(row["match"], row["cut_minute"]) for row in anchors}
    candidates = [row for row in candidates if (row["match"], row["cut_minute"]) not in excluded]

    rows_for_names = anchors + candidates
    names = feature_names(rows_for_names)
    anchor_x_raw = make_matrix(anchors, names)
    anchor_x, mu, sigma = standardize_from_reference(anchor_x_raw)
    candidate_x = (make_matrix(candidates, names) - mu) / sigma
    positive_vectors = [anchor_x[index] for index in range(len(positive_anchors))]
    negative_vectors = [anchor_x[index] for index in range(len(positive_anchors), len(anchors))]

    positive_summary = {key: positive_anchor[key] for key in ("n", "mean_delta_pp", "std_delta_pp", "lcb90_delta_pp", "cvar25_delta_pp", "worst_delta_pp", "positive_rate")}
    negative_summary = {key: negative_anchors[0][key] for key in ("n", "mean_delta_pp", "std_delta_pp", "lcb90_delta_pp", "cvar25_delta_pp", "worst_delta_pp", "positive_rate")} if negative_anchors else summarize_deltas([])
    expected_gap_pp = positive_summary["lcb90_delta_pp"] - negative_summary["mean_delta_pp"]

    for index, row in enumerate(candidates):
        vector = candidate_x[index]
        pos_distance = distance(vector, positive_vectors)
        neg_distance = distance(vector, negative_vectors)
        row["positive_anchor_distance"] = pos_distance
        row["negative_anchor_distance"] = neg_distance
        row["distance_margin"] = neg_distance - pos_distance
        row["compatibility_score"] = row["distance_margin"] - 0.10 * pos_distance
        row["estimated_lcb_pp"] = positive_summary["lcb90_delta_pp"] - 0.50 * pos_distance + 0.20 * row["distance_margin"]
        row["estimated_mean_pp"] = positive_summary["mean_delta_pp"] - 0.75 * pos_distance + 0.15 * expected_gap_pp

    candidates = sorted(
        candidates,
        key=lambda row: (row["compatibility_score"], row["estimated_lcb_pp"], -row["positive_anchor_distance"]),
        reverse=True,
    )
    eval_candidates = candidates[: args.eval_top_k]
    output = {
        "checkpoint": args.checkpoint,
        "action": args.action,
        "candidate_minutes": parse_minutes(args.candidate_minutes),
        "anchor_logic": "rank close to successful action-blend anchors and away from hard negative action-blend states",
        "positive_anchors": [{key: value for key, value in row.items() if key != "features"} for row in positive_anchors],
        "negative_anchors": [{key: value for key, value in row.items() if key != "features"} for row in negative_anchors],
        "positive_summary": positive_summary,
        "negative_summary": negative_summary,
        "candidate_count": len(candidates),
        "top_candidates": candidates[: args.top_k],
        "eval_candidates": eval_candidates,
        "recommended_eval": {
            "status": "ready" if eval_candidates else "no_candidates",
            "top_k": len(eval_candidates),
            "rollout_seeds": args.eval_rollout_seeds,
            "commands": [build_single_eval_command(args, row, index) for index, row in enumerate(eval_candidates, start=1)],
        },
        "feature_names": names,
    }
    save_json(output, args.out_json)
    print(json.dumps({
        "candidate_count": output["candidate_count"],
        "positive_lcb90_pp": positive_summary["lcb90_delta_pp"],
        "negative_mean_pp": negative_summary["mean_delta_pp"],
        "top_eval": [
            {
                "match": row["match"],
                "cut_minute": row["cut_minute"],
                "score": row["compatibility_score"],
                "pos_distance": row["positive_anchor_distance"],
                "neg_distance": row["negative_anchor_distance"],
                "estimated_lcb_pp": row["estimated_lcb_pp"],
            }
            for row in eval_candidates
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
