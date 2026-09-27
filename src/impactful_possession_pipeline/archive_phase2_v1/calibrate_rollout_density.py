import argparse
from collections import Counter

from .config import DataConfig, TokenConfig
from .dataset import frames_after_cut
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, mean, save_json
from .vocabulary import Vocab


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return float(xs[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real event-frame density on a checkpoint holdout split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/rollout_density.json")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all checkpoint test files.")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=60)
    args = parser.parse_args()

    _, vocab_obj, _, extra = load_checkpoint(args.checkpoint, "cpu")
    if "test_files" not in extra or "data_cfg" not in extra:
        raise RuntimeError("Checkpoint does not include test_files/data_cfg metadata.")

    vocab = Vocab.from_dict(vocab_obj)
    data_cfg = DataConfig(**extra["data_cfg"])
    encoder = StructuredStatsBombEncoder(vocab, TokenConfig())
    test_files = list(extra["test_files"])
    if args.max_files > 0:
        test_files = test_files[: args.max_files]

    horizon = max(1, args.end_minute - args.cut_minute)
    rows = []
    event_types = Counter()
    teams = Counter()
    for path in test_files:
        events = load_json(path)
        _, frames = encoder.encode_match(events, encoder.lineup_path_for_event_file(path, data_cfg.lineups_dir))
        real_frames = frames_after_cut(frames, args.cut_minute, horizon)
        for frame in real_frames:
            event_types[frame.get("event_type", "unknown")] += 1
            teams[frame.get("team", "unknown")] += 1
        rows.append({"match": path, "frames": len(real_frames), "events_per_minute": len(real_frames) / horizon})

    densities = [row["events_per_minute"] for row in rows]
    summary = {
        "checkpoint": args.checkpoint,
        "num_matches": len(rows),
        "cut_minute": args.cut_minute,
        "end_minute": args.end_minute,
        "mean_events_per_minute": mean(densities),
        "p10_events_per_minute": percentile(densities, 0.10),
        "p25_events_per_minute": percentile(densities, 0.25),
        "p50_events_per_minute": percentile(densities, 0.50),
        "p75_events_per_minute": percentile(densities, 0.75),
        "p90_events_per_minute": percentile(densities, 0.90),
        "recommended_min_events_per_minute": max(1, int(percentile(densities, 0.10))),
        "recommended_max_events_per_minute": max(1, int(percentile(densities, 0.90))),
        "event_type_counts": dict(event_types.most_common()),
        "team_counts": dict(teams.most_common()),
    }
    save_json({"summary": summary, "matches": rows}, args.out_json)
    print(f"saved rollout density calibration to {args.out_json}")
    print(summary)


if __name__ == "__main__":
    main()
