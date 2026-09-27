import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .config import DataConfig
from .statsbomb_features import get_home_team_id, load_lineup_roles, replacement_role, role_from_position
from .utils import list_event_files, load_json, split_files


def team_name(event: dict, home_team_id: int | None) -> str:
    return "HOME" if event.get("team", {}).get("id") == home_team_id else "AWAY"


def main() -> None:
    parser = argparse.ArgumentParser(description="Count support for richer intervention action tokens.")
    parser.add_argument("--max-matches", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    files = list_event_files(DataConfig.events_dir, args.max_matches, DataConfig.seed)
    train_files, val_files, test_files = split_files(files, DataConfig.train_fraction, DataConfig.val_fraction)
    splits = {"train": train_files, "val": val_files, "test": test_files}
    counts: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    role_pairs: Counter[tuple[str, str]] = Counter()
    formation_pairs: Counter[tuple[str, str]] = Counter()

    scanned = 0
    for split, paths in splits.items():
        for event_file in paths:
            scanned += 1
            if args.progress_every > 0 and scanned % args.progress_every == 0:
                print(f"[progress] scanned {scanned}/{len(files)}", flush=True)
            events = load_json(event_file)
            home_team_id = get_home_team_id(events)
            lineup_roles = load_lineup_roles(str(Path(DataConfig.lineups_dir) / Path(event_file).name))
            current_formation = {"HOME": "433", "AWAY": "433"}
            ordered = sorted(
                events,
                key=lambda event: (
                    event.get("period", 1),
                    event.get("minute", 0),
                    event.get("second", 0),
                    event.get("index", 0),
                ),
            )
            for event in ordered:
                event_type = event.get("type", {}).get("name")
                team = team_name(event, home_team_id)
                if event_type == "Starting XI":
                    current_formation[team] = str(event.get("tactics", {}).get("formation", current_formation[team]))
                    continue
                if event_type == "Substitution":
                    outgoing = role_from_position(event.get("position", {}).get("name", ""))
                    incoming = replacement_role(event, lineup_roles)
                    token = f"ACTION::{team}_SUB_{outgoing.upper()}_TO_{incoming.upper()}"
                    counts[token] += 1
                    by_split[split][token] += 1
                    role_pairs[(outgoing, incoming)] += 1
                    continue
                if event_type == "Tactical Shift":
                    old = str(current_formation.get(team, "433"))
                    new = str(event.get("tactics", {}).get("formation", old))
                    token = f"ACTION::{team}_TACTIC_{old}_TO_{new}"
                    counts[token] += 1
                    by_split[split][token] += 1
                    formation_pairs[(old, new)] += 1
                    current_formation[team] = new

    print(f"files train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"unique_rich_action_tokens={len(counts)}")
    print("top_rich_tokens")
    for token, total in counts.most_common(60):
        print(
            total,
            token,
            f"train={by_split['train'][token]}",
            f"val={by_split['val'][token]}",
            f"test={by_split['test'][token]}",
        )
    print("top_role_pairs")
    for pair, total in role_pairs.most_common(25):
        print(total, pair)
    print("top_formation_pairs")
    for pair, total in formation_pairs.most_common(25):
        print(total, pair)


if __name__ == "__main__":
    main()
