import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


POSSESSION_EVENT_TYPES = {
    "Pass",
    "Carry",
    "Ball Receipt*",
    "Ball Recovery",
    "Interception",
    "Shot",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def list_event_files(events_dir: str, max_matches: int, seed: int) -> List[str]:
    paths = [str(Path(events_dir) / name) for name in os.listdir(events_dir) if name.endswith(".json")]
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:max_matches]


def split_files(paths: Sequence[str], train_fraction: float, val_fraction: float):
    n = len(paths)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    train = list(paths[:n_train])
    val = list(paths[n_train:n_train + n_val])
    test = list(paths[n_train + n_val:])
    return train, val, test


def mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else default


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def event_type_counts(frames: Iterable[dict]) -> dict:
    """Count real match events by type, excluding injected Action/state tokens.

    Mirrors the paper's Table 1 columns: per-type counts feed Passes/Shots/
    Dribbles MAD, and the sum over all real event types feeds Total Events MAD.
    """
    counts: dict = {}
    for frame in frames:
        event_type = frame.get("event_type")
        if event_type is None or event_type in {"Action", "Half Start", "Half End"}:
            continue
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def home_possession_ratio(frames: Iterable[dict]) -> float:
    home = 0
    total = 0
    for frame in frames:
        if frame.get("event_type") in POSSESSION_EVENT_TYPES:
            team = frame.get("team")
            if team in {"home", "away"}:
                total += 1
                if team == "home":
                    home += 1
    return home / total if total else 0.5


def home_shot_ratio(frames: Iterable[dict]) -> float:
    """Home share of generated shots, on the same 0--1 scale as possession."""
    home = 0
    total = 0
    for frame in frames:
        if frame.get("event_type") == "Shot" and frame.get("team") in {"home", "away"}:
            total += 1
            if frame.get("team") == "home":
                home += 1
    return home / total if total else 0.5


def home_shot_count(frames: Iterable[dict]) -> int:
    return sum(
        1 for frame in frames
        if frame.get("event_type") == "Shot" and frame.get("team") == "home"
    )


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def bootstrap_mean_lcb(values: Sequence[float], alpha: float = 0.10, num_resamples: int = 2000, seed: int = 0) -> float:
    """Percentile-bootstrap lower confidence bound on the mean.

    Used to gate any "this intervention helps" claim: require this LCB > 0
    across the full pinned holdout, instead of eyeballing a handful of
    single-match rollout seeds (the pattern that produced and then retracted
    the 2026-07-18 Phase 2 "corridor milestones").
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rng = random.Random(seed)
    n = len(values)
    resample_means = []
    for _ in range(num_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(sample) / n)
    resample_means.sort()
    idx = int(alpha * num_resamples)
    return resample_means[max(0, min(len(resample_means) - 1, idx))]
