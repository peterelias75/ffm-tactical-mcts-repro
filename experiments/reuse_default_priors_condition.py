"""Reuse one completed default priors-MCTS result in identical sweep points."""

import argparse
import hashlib
import shutil
from pathlib import Path

from impactful_possession_pipeline.utils import load_json, save_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    data = load_json(source)
    summary = data.get("summary", {})
    required = {
        "complete": True,
        "num_matches": 200,
        "num_requested_matches": 200,
        "num_simulations": 10,
        "step_minutes": 5,
        "uniform_prior_weight": 0.05,
    }
    mismatches = {key: {"expected": value, "observed": summary.get(key)} for key, value in required.items() if summary.get(key) != value}
    if summary.get("rollout_lookahead_tokens") not in (None, 200):
        mismatches["rollout_lookahead_tokens"] = {"expected": "None/default 5 minutes or explicit 200", "observed": summary.get("rollout_lookahead_tokens")}
    if mismatches:
        raise RuntimeError(f"source is not the completed default priors condition: {mismatches}")
    digest = sha256(source)
    targets = []
    for target_text in args.target:
        target = Path(target_text)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise RuntimeError(f"hash mismatch after copying {target}")
        targets.append(str(target))
    save_json({
        "source": str(source), "source_sha256": digest, "targets": targets,
        "identity": {
            "E06_step_5": "default lookahead equals step_minutes=5",
            "E07_num_simulations_10": "default priors, step_minutes=5, num_simulations=10",
            "E08_lookahead_tokens_200": "200 tokens / 40 tokens-per-minute = 5-minute default lookahead",
        },
    }, args.manifest)
    print(f"validated and copied identical condition to {len(targets)} targets")


if __name__ == "__main__":
    main()
