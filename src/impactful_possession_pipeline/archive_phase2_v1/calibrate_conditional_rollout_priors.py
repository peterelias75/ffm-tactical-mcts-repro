import argparse
from collections import Counter, defaultdict

import torch

from .config import DataConfig, TokenConfig
from .dataset import frames_after_cut
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import load_json, save_json
from .vocabulary import EVENT_TYPES, PLAYER_ROLES, Vocab


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalize(counter: Counter, support, alpha: float) -> dict:
    smoothed = Counter(counter)
    for key in support:
        smoothed[str(key)] += alpha
    total = sum(smoothed.values()) or 1.0
    return {str(key): float(smoothed[str(key)]) / total for key in support}


def normalize_nested(counters: dict, support, alpha: float, min_count: int) -> dict:
    out = {}
    for key, counter in counters.items():
        if sum(counter.values()) >= min_count:
            out[str(key)] = normalize(counter, support, alpha)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate conditional holdout priors for rollout realism.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all checkpoint test files.")
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=50)
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

    event_given_prev = defaultdict(Counter)
    role_given_event = defaultdict(Counter)
    lx_given_event = defaultdict(Counter)
    ly_given_event = defaultdict(Counter)
    event_counts = Counter()
    frame_count = 0

    for event_file in files:
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, data_cfg.lineups_dir)
        _, frames = encoder.encode_match(events, lineup_path)
        previous_event = "<START>"
        for frame in frames_after_cut(frames, args.cut_minute, args.end_minute - args.cut_minute):
            event_type = frame.get("event_type")
            if event_type == "Action" or event_type not in EVENT_TYPES:
                continue
            frame_count += 1
            event_counts[event_type] += 1
            event_given_prev[previous_event][event_type] += 1
            previous_event = event_type
            role = frame.get("role")
            if role in PLAYER_ROLES:
                role_given_event[event_type][role] += 1
            if "lx" in frame:
                lx_given_event[event_type][str(frame["lx"])] += 1
            if "ly" in frame:
                ly_given_event[event_type][str(frame["ly"])] += 1

    obj = {
        "checkpoint": args.checkpoint,
        "holdout_files": len(files),
        "cut_minute": args.cut_minute,
        "end_minute": args.end_minute,
        "frame_count": frame_count,
        "alpha": args.alpha,
        "min_count": args.min_count,
        "event_type_given_prev": normalize_nested(event_given_prev, EVENT_TYPES, args.alpha, args.min_count),
        "role_given_event": normalize_nested(role_given_event, PLAYER_ROLES, args.alpha, args.min_count),
        "lx_given_event": normalize_nested(lx_given_event, [str(i) for i in range(token_cfg.x_bins)], args.alpha, args.min_count),
        "ly_given_event": normalize_nested(ly_given_event, [str(i) for i in range(token_cfg.y_bins)], args.alpha, args.min_count),
        "raw_event_counts": dict(event_counts.most_common()),
        "condition_counts": {
            "event_type_given_prev": {str(k): sum(v.values()) for k, v in event_given_prev.items()},
            "role_given_event": {str(k): sum(v.values()) for k, v in role_given_event.items()},
            "lx_given_event": {str(k): sum(v.values()) for k, v in lx_given_event.items()},
            "ly_given_event": {str(k): sum(v.values()) for k, v in ly_given_event.items()},
        },
    }
    save_json(obj, args.out_json)
    print(f"saved conditional rollout priors to {args.out_json}")
    print({"holdout_files": len(files), "frame_count": frame_count, "event_conditions": len(obj["event_type_given_prev"]), "top_events": event_counts.most_common(8)})


if __name__ == "__main__":
    main()
