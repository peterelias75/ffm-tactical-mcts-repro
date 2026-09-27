import argparse
import math
from pathlib import Path

import torch

from .config import DataConfig
from .select_robust_intervention_candidates import make_feature_names, make_matrix, standardize, apply_standardize
from .train_context_intervention_scorer import checkpoint_event_map, context_features
from .utils import load_json, save_json


def checkpoint_test_files(checkpoint: str) -> list[str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    return list(ckpt.get("extra", {}).get("test_files", []))


def parse_minutes(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def euclidean(a, b) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank match/minute states by similarity to a robust intervention anchor.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor-match-json", required=True)
    parser.add_argument("--anchor-minute", type=int, default=45)
    parser.add_argument("--action", default="HOME_SUB_ROLE_CENTRAL_MIDFIELDER")
    parser.add_argument("--candidate-minutes", default="40,45,50,55")
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--max-matches", type=int, default=200)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    event_by_name = checkpoint_event_map(args.checkpoint, args.events_dir or DataConfig.events_dir)
    test_files = checkpoint_test_files(args.checkpoint)[: args.max_matches]
    minutes = parse_minutes(args.candidate_minutes)
    action = args.action.replace("ACTION::", "")
    base_row = {
        "value_head_delta_pp": 0.0,
        "mean_noop_home_possession": 0.5,
        "action": action,
    }

    anchor_events = load_json(args.anchor_match_json)
    anchor_feats = context_features(anchor_events, action, base_row, args.anchor_minute)
    feature_rows = [{"features": anchor_feats}]
    candidate_raw = []
    for match_json in test_files:
        path = event_by_name.get(Path(match_json).name, match_json)
        if not Path(path).exists():
            continue
        events = load_json(path)
        for minute in minutes:
            feats = context_features(events, action, base_row, minute)
            row = {
                "match": Path(path).name,
                "match_json": path,
                "cut_minute": minute,
                "features": feats,
            }
            feature_rows.append(row)
            candidate_raw.append(row)

    feature_names = make_feature_names(feature_rows)
    matrix = make_matrix(feature_rows, feature_names)
    x, mu, sigma = standardize(matrix)
    anchor_vec = x[0]
    candidates = []
    for raw, vector in zip(candidate_raw, x[1:]):
        if raw["match"] == Path(args.anchor_match_json).name and raw["cut_minute"] == args.anchor_minute:
            continue
        distance = euclidean(vector, anchor_vec)
        feats = raw["features"]
        candidates.append(
            {
                "match": raw["match"],
                "match_json": raw["match_json"],
                "cut_minute": raw["cut_minute"],
                "end_minute": raw["cut_minute"] + 5,
                "action": f"ACTION::{action}",
                "anchor_distance": distance,
                "score_diff": feats.get("score_diff"),
                "recent15_home_possession": feats.get("recent15_home_possession"),
                "recent15_events_per_min": feats.get("recent15_events_per_min"),
                "home_recent15_event_share": feats.get("home_recent15_event_share"),
                "home_recent15_final_third_per_min": feats.get("home_recent15_final_third_per_min"),
                "away_recent15_final_third_per_min": feats.get("away_recent15_final_third_per_min"),
            }
        )
    candidates = sorted(candidates, key=lambda row: row["anchor_distance"])
    save_json(
        {
            "checkpoint": args.checkpoint,
            "anchor": {
                "match_json": args.anchor_match_json,
                "cut_minute": args.anchor_minute,
                "action": f"ACTION::{action}",
                "features": anchor_feats,
            },
            "candidate_minutes": minutes,
            "scanned_matches": len(test_files),
            "feature_names": feature_names,
            "top_candidates": candidates[: args.top_k],
            "all_candidates": candidates,
        },
        args.out_json,
    )
    print(f"saved {len(candidates)} timing candidates to {args.out_json}")
    for row in candidates[: args.top_k]:
        print(f"{row['match']} minute={row['cut_minute']} distance={row['anchor_distance']:.3f}")


if __name__ == "__main__":
    main()
