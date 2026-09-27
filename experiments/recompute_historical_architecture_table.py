"""Recompute the manuscript architecture table from the surviving raw CSV artifacts."""

import hashlib
from pathlib import Path

from impactful_possession_pipeline.utils import mean, save_json
from paper_reproduction.recompute_historical_e01 import BALL_EVENTS, parse_sections


PAPER = {
    "RNN": [14.71, 67.40, 6.38, 4.04, 13.69],
    "GNN message passing": [13.08, 58.78, 8.62, 4.14, 11.80],
    "DistilGPT2": [12.85, 63.23, 4.36, 4.84, 9.65],
    "GPT2 Small": [13.82, 63.63, 4.51, 6.43, 10.25],
    "GPT-Neo": [11.90, 90.41, 5.53, 7.61, 13.66],
    "GPT2 Medium": [11.17, 58.38, 3.33, 4.84, 10.54],
    "OPT350": [11.24, 114.58, 3.86, 33.13, 20.68],
    "FFM": [11.95, 71.27, 4.34, 5.52, 11.04],
}

# Reproduce the selections literally present in each surviving aggregator.
SPECS = {
    "RNN": (Path("rnn/results"), "*_results", lambda xs: xs[:450][5:], "sorted[:450], then [5:]"),
    "GNN message passing": (Path("gnn_message_passing/results"), "*_results", lambda xs: xs[5:], "sorted all, then [5:] (source contains a syntax-damaged [] token)"),
    "DistilGPT2": (Path("gpt_dist/gpt_tagged_subs/results"), "*_results", lambda xs: xs[70:520], "sorted[70:520]"),
    "GPT2 Small": (Path("gpt_small/gpt_tagged_subs/results"), "*_results", lambda xs: xs[70:520], "filesystem-order[70:520] in source; sorted used here for deterministic audit"),
    "GPT-Neo": (Path("gpt_neo/gpt_tagged_subs/results"), "*_results", lambda xs: xs[70:520], "sorted[70:520]"),
    "GPT2 Medium": (Path("gpt_medium/results"), "*results_small_grid", lambda xs: xs[5:], "sorted all, then [5:]"),
    "OPT350": (Path("opt_350/opt_tagged_subs/results"), "*_results", lambda xs: xs[70:520], "sorted[70:520]"),
    "FFM": (Path("model_v3/results"), "*_results", lambda xs: xs[70:520], "sorted[70:520]"),
}


def aggregate(items):
    keys = sorted({key for item in items for key in item})
    return {key: mean([item.get(key, {}).get("Difference", 0.0) for item in items]) for key in keys}


def compute(name, root, pattern, selector, selection):
    available = sorted(path for path in root.glob(pattern) if path.is_dir())
    selected = selector(available)
    homes, aways, possession, digest, skipped = [], [], [], hashlib.sha256(), []
    for directory in selected:
        event_path = directory / "events.csv"
        if not event_path.is_file():
            skipped.append(directory.name)
            continue
        digest.update(str(event_path.relative_to(root)).encode()); digest.update(event_path.read_bytes())
        home, away = parse_sections(event_path)
        homes.append(home); aways.append(away)
        gt_h = sum(v["GT"] for k, v in home.items() if k.lower() in BALL_EVENTS)
        gt_a = sum(v["GT"] for k, v in away.items() if k.lower() in BALL_EVENTS)
        ge_h = sum(v["GEN"] for k, v in home.items() if k.lower() in BALL_EVENTS)
        ge_a = sum(v["GEN"] for k, v in away.items() if k.lower() in BALL_EVENTS)
        possession.append(abs(100 * gt_h / (gt_h + gt_a) - 100 * ge_h / (ge_h + ge_a)))
    eh, ea = aggregate(homes), aggregate(aways)
    observed = [mean(possession), mean([eh.get("pass", 0), ea.get("pass", 0)]),
                mean([eh.get("shot", 0), ea.get("shot", 0)]),
                mean([eh.get("dribble", 0), ea.get("dribble", 0)]),
                mean(list(eh.values()) + list(ea.values()))]
    paper = PAPER[name]
    return {"results_dir": str(root), "available_directories": len(available), "selection": selection,
            "selected_directories": len(selected), "complete_event_csvs": len(homes), "skipped": skipped,
            "event_csv_sha256": digest.hexdigest(),
            "observed": dict(zip(["possession_mae_pp", "passes_mae", "shots_mae", "dribbles_mae", "total_events_category_average"], observed)),
            "paper": dict(zip(["possession_mae_pp", "passes_mae", "shots_mae", "dribbles_mae", "total_events_mae"], paper)),
            "rounded_exact": [round(a, 2) == b for a, b in zip(observed, paper)],
            "home_event_mad": eh, "away_event_mad": ea}


def main():
    rows = {name: compute(name, *spec) for name, spec in SPECS.items()}
    result = {"experiment_id": "E01_architecture_table_historical_audit", "paper_n": 450,
              "metric_caveat": "The historical total is a category-average MAD, not aggregate event-count MAE.", "rows": rows}
    save_json(result, "paper_reproduction/runs/e01_architecture_table/results.json")
    for name, row in rows.items():
        print(name, row["complete_event_csvs"], row["observed"], row["rounded_exact"])


if __name__ == "__main__":
    main()
