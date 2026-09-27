import time
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, load_json


IGNORE_INDEX = -100


def token_realism_weight(token: str, weights: dict | None = None) -> float:
    weights = weights or {}
    if token.startswith("EVT::"):
        return float(weights.get("event", 1.0))
    if token.startswith("TEAM::"):
        return float(weights.get("team", 1.0))
    if token.startswith("ROLE::"):
        return float(weights.get("role", 1.0))
    if token.startswith("LX::") or token.startswith("LY::"):
        return float(weights.get("location", 1.0))
    if token.startswith("PFLIP::"):
        return float(weights.get("possession_flip", 1.0))
    if token.startswith("GOAL::"):
        return float(weights.get("goal", 1.0))
    return 1.0


def frames_after_cut(frames: Sequence[dict], cut_minute: int, horizon_minutes: int | None = None) -> List[dict]:
    end = 10_000 if horizon_minutes is None else cut_minute + horizon_minutes
    return [f for f in frames if cut_minute <= int(f.get("minute", 0)) < end]


def frame_token_width(frame: dict) -> int:
    # Frames do not include <bos>. Action/State frames are a single injected
    # token; regular event frames are the full 9-field + <sep> cycle.
    return int(frame.get("width", 10))


def token_index_at_cut(frames: Sequence[dict], cut_minute: int) -> int:
    idx = 1
    for frame in frames:
        if int(frame.get("minute", 0)) >= cut_minute:
            break
        idx += frame_token_width(frame)
    return idx


def action_anchor_positions(frames: Sequence[dict]) -> List[tuple[int, int]]:
    """Token index just before/after each real (organic) ACTION:: event.

    Real substitutions and tactical shifts are rare relative to the fixed
    cut-minute grid, so the value head sees very few examples where an action
    token sits right at the edge of its context. Anchoring extra examples
    directly to these real events (rather than fabricating hypothetical ones)
    increases the density of "context ends right at/after a real action"
    training signal without inventing any labels: the target is still the
    real observed future in both cases. This does not by itself resolve
    confounding-by-indication (managers sub for reasons correlated with the
    game state) -- it only gives the model more real examples to learn from
    around the moments actions actually happened.
    """
    positions = []
    idx = 1
    for frame in frames:
        if frame.get("event_type") == "Action":
            positions.append((idx, int(frame.get("minute", 0))))
        idx += frame_token_width(frame)
    return positions


class RandomCutForwardDataset(Dataset):
    def __init__(
        self,
        event_files: Sequence[str],
        lineups_dir: str | None,
        encoder: StructuredStatsBombEncoder,
        data_cfg: DataConfig,
        max_len: int,
        label_weights: dict | None = None,
        progress_every: int = 0,
        progress_label: str = "",
    ):
        self.examples = []
        self.encoder = encoder
        self.max_len = max_len
        self.label_weights = label_weights or {}
        start_time = time.time()
        for file_idx, event_file in enumerate(event_files):
            if progress_every and file_idx % progress_every == 0:
                elapsed = time.time() - start_time
                rate = file_idx / elapsed if elapsed > 0 else 0.0
                remaining = (len(event_files) - file_idx) / rate if rate > 0 else float("nan")
                print(
                    f"  [{progress_label}] {file_idx}/{len(event_files)} matches encoded "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, examples so far={len(self.examples)})",
                    flush=True,
                )
            events = load_json(event_file)
            lineup_path = encoder.lineup_path_for_event_file(event_file, lineups_dir)
            ids, frames = encoder.encode_match(events, lineup_path)
            seen_cut_idx = set()
            for cut_minute in data_cfg.cut_minutes[:data_cfg.max_examples_per_match]:
                cut_idx = token_index_at_cut(frames, cut_minute)
                if cut_idx <= 2 or cut_idx >= len(ids) - 2:
                    continue
                future = frames_after_cut(frames, cut_minute)
                if not future:
                    continue
                seen_cut_idx.add(cut_idx)
                self.examples.append({
                    "ids": ids,
                    "cut_idx": cut_idx,
                    "cut_minute": cut_minute,
                    "target_pos_5": home_possession_ratio(frames_after_cut(frames, cut_minute, 5)),
                    "target_pos_10": home_possession_ratio(frames_after_cut(frames, cut_minute, 10)),
                    "target_pos_rest": home_possession_ratio(future),
                    "target_count_5": float(len(frames_after_cut(frames, cut_minute, 5))),
                    "target_count_10": float(len(frames_after_cut(frames, cut_minute, 10))),
                    "path": event_file,
                    "anchor": "grid",
                })
            if getattr(data_cfg, "action_anchor_pairs", False):
                for pre_idx, action_minute in action_anchor_positions(frames):
                    for anchor_name, cut_idx in (("pre_action", pre_idx), ("post_action", pre_idx + 1)):
                        if cut_idx <= 2 or cut_idx >= len(ids) - 2 or cut_idx in seen_cut_idx:
                            continue
                        future = frames_after_cut(frames, action_minute)
                        if not future:
                            continue
                        seen_cut_idx.add(cut_idx)
                        self.examples.append({
                            "ids": ids,
                            "cut_idx": cut_idx,
                            "cut_minute": action_minute,
                            "target_pos_5": home_possession_ratio(frames_after_cut(frames, action_minute, 5)),
                            "target_pos_10": home_possession_ratio(frames_after_cut(frames, action_minute, 10)),
                            "target_pos_rest": home_possession_ratio(future),
                            "target_count_5": float(len(frames_after_cut(frames, action_minute, 5))),
                            "target_count_10": float(len(frames_after_cut(frames, action_minute, 10))),
                            "path": event_file,
                            "anchor": anchor_name,
                        })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        ids = ex["ids"]
        cut_idx = ex["cut_idx"]
        if len(ids) > self.max_len:
            token_budget = self.max_len - 1
            context_budget = max(1, token_budget // 2)
            start = max(1, cut_idx - context_budget)
            end = min(len(ids), start + token_budget)
            if cut_idx >= end:
                start = max(1, cut_idx - token_budget + 1)
                end = min(len(ids), start + token_budget)
            ids = [self.encoder.bos_id] + ids[start:end]
            cut_idx = cut_idx - start + 1
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        labels = y.clone()
        labels[:max(0, cut_idx - 1)] = IGNORE_INDEX
        label_weights = torch.ones_like(y, dtype=torch.float32)
        for pos, token_id in enumerate(y.tolist()):
            if labels[pos].item() != IGNORE_INDEX:
                label_weights[pos] = token_realism_weight(self.encoder.vocab.token(token_id), self.label_weights)
        context = torch.tensor(ids[:cut_idx], dtype=torch.long)
        return {
            "x": x,
            "labels": labels,
            "label_weights": label_weights,
            "context": context,
            "target_pos_5": torch.tensor(ex["target_pos_5"], dtype=torch.float32),
            "target_pos_10": torch.tensor(ex["target_pos_10"], dtype=torch.float32),
            "target_pos_rest": torch.tensor(ex["target_pos_rest"], dtype=torch.float32),
            "target_count_5": torch.tensor(ex["target_count_5"], dtype=torch.float32),
            "target_count_10": torch.tensor(ex["target_count_10"], dtype=torch.float32),
            "cut_minute": ex["cut_minute"],
            "path": ex["path"],
            "anchor": ex.get("anchor", "grid"),
        }


def collate_forward(batch: Sequence[dict], pad_id: int) -> dict:
    max_x = max(item["x"].numel() for item in batch)
    max_ctx = max(item["context"].numel() for item in batch)
    x = torch.full((len(batch), max_x), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_x), IGNORE_INDEX, dtype=torch.long)
    label_weights = torch.ones((len(batch), max_x), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_x), dtype=torch.bool)
    context = torch.full((len(batch), max_ctx), pad_id, dtype=torch.long)
    context_mask = torch.zeros((len(batch), max_ctx), dtype=torch.bool)
    for i, item in enumerate(batch):
        nx = item["x"].numel()
        nc = item["context"].numel()
        x[i, :nx] = item["x"]
        labels[i, :nx] = item["labels"]
        label_weights[i, :nx] = item["label_weights"]
        mask[i, :nx] = True
        context[i, :nc] = item["context"]
        context_mask[i, :nc] = True
    return {
        "x": x,
        "labels": labels,
        "label_weights": label_weights,
        "mask": mask,
        "context": context,
        "context_mask": context_mask,
        "target_pos_5": torch.stack([item["target_pos_5"] for item in batch]),
        "target_pos_10": torch.stack([item["target_pos_10"] for item in batch]),
        "target_pos_rest": torch.stack([item["target_pos_rest"] for item in batch]),
        "target_count_5": torch.stack([item["target_count_5"] for item in batch]),
        "target_count_10": torch.stack([item["target_count_10"] for item in batch]),
        "paths": [item["path"] for item in batch],
    }
