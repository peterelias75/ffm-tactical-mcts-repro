import argparse
from collections import Counter
from pathlib import Path

import torch

from .config import DataConfig, TokenConfig
from .dataset import frames_after_cut
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, save_json
from .vocabulary import Vocab


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalize(counter: Counter, smoothing_keys=(), alpha: float = 1.0) -> dict:
    smoothed = Counter(counter)
    for key in smoothing_keys:
        smoothed[str(key)] += alpha
    total = sum(smoothed.values()) or 1.0
    return {str(key): float(value) / total for key, value in sorted(smoothed.items(), key=lambda item: str(item[0]))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate real holdout priors for rollout event realism.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all checkpoint test files.")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _, vocab_obj, _, extra = load_checkpoint(args.checkpoint, resolve_device(args.device))
    if "test_files" not in extra or "data_cfg" not in extra:
        raise RuntimeError("Checkpoint does not include test_files/data_cfg metadata.")

    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig()
    data_cfg = DataConfig(**extra["data_cfg"])
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    files = list(extra["test_files"])
    if args.max_files > 0:
        files = files[: args.max_files]

    event_type = Counter()
    role = Counter()
    lx = Counter()
    ly = Counter()
    locations = Counter()
    frame_count = 0
    for event_file in files:
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, data_cfg.lineups_dir)
        _, frames = encoder.encode_match(events, lineup_path)
        for frame in frames_after_cut(frames, args.cut_minute, args.end_minute - args.cut_minute):
            if frame.get("event_type") == "Action":
                continue
            frame_count += 1
            event_type[frame.get("event_type", "unknown")] += 1
            role[frame.get("role", "unknown")] += 1
            if "lx" in frame:
                lx[str(frame["lx"])] += 1
            if "ly" in frame:
                ly[str(frame["ly"])] += 1
            if "lx" in frame and "ly" in frame:
                locations[f"{frame['lx']}:{frame['ly']}"] += 1

    obj = {
        "checkpoint": args.checkpoint,
        "holdout_files": len(files),
        "cut_minute": args.cut_minute,
        "end_minute": args.end_minute,
        "frame_count": frame_count,
        "event_type": normalize(event_type),
        "role": normalize(role),
        "lx": normalize(lx, range(token_cfg.x_bins)),
        "ly": normalize(ly, range(token_cfg.y_bins)),
        "location": normalize(locations),
        "raw_counts": {
            "event_type": dict(event_type.most_common()),
            "role": dict(role.most_common()),
            "lx": dict(lx.most_common()),
            "ly": dict(ly.most_common()),
        },
    }
    save_json(obj, args.out_json)
    print(f"saved rollout priors to {args.out_json}")
    print({"holdout_files": len(files), "frame_count": frame_count, "top_event_types": event_type.most_common(8)})


if __name__ == "__main__":
    main()
