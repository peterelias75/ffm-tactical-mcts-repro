"""Classic Markov baseline: a fixed (non-learned, non-search) action-value table.

Mines real historical manager decisions (organic ACTION::HOME_* substitution and
tactical-shift tokens already present in ground-truth encoded matches -- see
statsbomb_features.encode_match) and estimates, for each action type, the mean
change in home possession ratio in the `step_minutes` window right after the
action versus right before it. This is a first-order empirical transition table:
P(possession outcome | action type), fixed once from data and never updated
adaptively at evaluation time -- in contrast to the MCTS policies, which search.

At decision time, `classic_markov_decide` just looks up the table and greedily
picks the best *currently legal* action, with no rollout and no per-match
adaptation beyond the legality constraints (max subs/tactics, cooldown) shared
by every policy in this comparison.
"""

import argparse
from pathlib import Path
from typing import Sequence

from .config import DataConfig, PlannerConfig, TokenConfig
from .planner import legal_actions
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, list_event_files, load_json, mean, save_json, split_files
from .vocabulary import Vocab, build_vocab


def _possession_window(frames: Sequence[dict], lo: float, hi: float) -> float:
    windowed = [f for f in frames if lo <= float(f.get("minute", 0)) < hi]
    return home_possession_ratio(windowed)


def build_transition_table(
    event_files: Sequence[str],
    lineups_dir: str,
    vocab: Vocab,
    token_cfg: TokenConfig,
    step_minutes: int = 5,
) -> dict:
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    deltas_by_action: dict = {}
    noop_deltas: list = []

    for path in event_files:
        try:
            events = load_json(path)
        except Exception:
            continue
        lineup_path = encoder.lineup_path_for_event_file(path, lineups_dir)
        try:
            _, frames = encoder.encode_match(events, lineup_path)
        except Exception:
            continue

        action_minutes = [
            (f["minute"], f["action"])
            for f in frames
            if f.get("event_type") == "Action" and f.get("team") == "home" and str(f.get("action", "")).startswith("ACTION::HOME_")
        ]
        for minute, action in action_minutes:
            before = _possession_window(frames, minute - step_minutes, minute)
            after = _possession_window(frames, minute, minute + step_minutes)
            delta = after - before
            deltas_by_action.setdefault(action, []).append(delta)

        # NOOP reference: sample every step_minutes grid point that has no
        # organic action within the window, same before/after comparison.
        action_minute_set = {m for m, _ in action_minutes}
        for minute in range(15, 90, step_minutes):
            if any(abs(minute - m) < step_minutes for m in action_minute_set):
                continue
            before = _possession_window(frames, minute - step_minutes, minute)
            after = _possession_window(frames, minute, minute + step_minutes)
            noop_deltas.append(after - before)

    table = {action: mean(deltas) for action, deltas in deltas_by_action.items()}
    table["ACTION::NOOP"] = mean(noop_deltas)
    counts = {action: len(deltas) for action, deltas in deltas_by_action.items()}
    counts["ACTION::NOOP"] = len(noop_deltas)
    return {"mean_delta": table, "n_samples": counts, "step_minutes": step_minutes}


def classic_markov_decide(table: dict, legal: Sequence[str]) -> str:
    """Deterministic, fixed-table greedy choice among currently legal actions."""
    mean_delta = table["mean_delta"]
    default = mean_delta.get("ACTION::NOOP", 0.0)
    return max(legal, key=lambda a: mean_delta.get(a, default))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--max-matches", type=int, default=DataConfig.max_matches)
    parser.add_argument("--seed", type=int, default=DataConfig.seed)
    parser.add_argument("--step-minutes", type=int, default=PlannerConfig.step_minutes)
    parser.add_argument("--exclude-holdout-json", default=None, help="Match IDs (basenames) to exclude, e.g. the E05 holdout, so the table is never fit on evaluation matches.")
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/classic_markov_table.json")
    args = parser.parse_args()

    data_cfg = DataConfig()
    token_cfg = TokenConfig()
    vocab = build_vocab(token_cfg)

    files = list_event_files(args.events_dir, args.max_matches, args.seed)
    train_files, val_files, test_files = split_files(files, data_cfg.train_fraction, data_cfg.val_fraction)
    fit_files = train_files + val_files  # never touch the held-out test split

    if args.exclude_holdout_json:
        holdout = load_json(args.exclude_holdout_json)
        excluded = {str(m) for m in holdout.get("match_ids", [])}
        fit_files = [p for p in fit_files if Path(p).name not in excluded]

    table = build_transition_table(fit_files, args.lineups_dir, vocab, token_cfg, args.step_minutes)
    table["num_matches_scanned"] = len(fit_files)
    save_json(table, args.out_json)
    print(f"saved classic Markov transition table to {args.out_json}")
    print({"mean_delta": table["mean_delta"], "n_samples": table["n_samples"]})


if __name__ == "__main__":
    main()
