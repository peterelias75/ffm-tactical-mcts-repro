"""Parse historical random/Markov/MCTS/prior policy stdout into JSON.

The old scripts printed rounded percentages rather than machine-readable rows.
This parser preserves that limitation and reports both plausible meanings of
"gain": direct possession delta and reduction in absolute error versus reality.
"""

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

from impactful_possession_pipeline.utils import mean, save_json


PROCESS_RE = re.compile(r"(?:\[\d+/\d+\]\s+)?Processing\s+(.+?\.json)\s*$")
VALUE_RES = {
    "real": re.compile(r"^\s*Real(?: possession)?:\s+([0-9.]+)%\s*$", re.I),
    "noop": re.compile(r"^\s*No[- ]intervention(?: sim)?:\s+([0-9.]+)%\s*$", re.I),
    "policy": re.compile(r"^\s*With[- ]intervention(?: sim)?:\s+([0-9.]+)%\s*$", re.I),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows: list[dict]) -> dict:
    raw = [row["raw_possession_delta_pp"] for row in rows]
    error = [row["absolute_error_reduction_pp"] for row in rows]

    def metrics(values: list[float]) -> dict:
        positive = [value for value in values if value > 0]
        return {
            "n": len(values),
            "mean_gain_pp": mean(values),
            "pct_improved": 100.0 * mean([value > 0 for value in values]),
            "mean_gain_when_improved_pp": mean(positive) if positive else 0.0,
        }

    return {"raw_possession_gain": metrics(raw), "absolute_error_reduction": metrics(error)}


def parse(path: Path) -> list[dict]:
    rows, current = [], None
    for line in path.read_text(errors="replace").splitlines():
        match = PROCESS_RE.search(line)
        if match:
            if current and all(key in current for key in ("real_pp", "noop_pp", "policy_pp")):
                rows.append(current)
            current = {"match": Path(match.group(1)).name}
            continue
        if current is None:
            continue
        for key, regex in VALUE_RES.items():
            match = regex.match(line)
            if match:
                current[{"real": "real_pp", "noop": "noop_pp", "policy": "policy_pp"}[key]] = float(match.group(1))
                break
    if current and all(key in current for key in ("real_pp", "noop_pp", "policy_pp")):
        rows.append(current)
    for row in rows:
        row["raw_possession_delta_pp"] = row["policy_pp"] - row["noop_pp"]
        row["absolute_error_reduction_pp"] = abs(row["real_pp"] - row["noop_pp"]) - abs(row["real_pp"] - row["policy_pp"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="paper_reproduction/runs/e05_historical_policy_audit/results.json")
    args = parser.parse_args()
    sources = {
        "random": (Path("gpt_medium/random_guess.txt"), Path("gpt_medium/random_guess.py")),
        "classic_markov_historical": (Path("gpt_medium/markov.txt"), Path("gpt_medium/markov.py")),
        "plain_mcts_historical": (Path("gpt_medium/mcts.txt"), Path("gpt_medium/mcts.py")),
        "priors_historical": (Path("gpt_medium/priors.txt"), Path("gpt_medium/priors.py")),
    }
    variants = {}
    for name, (log, script) in sources.items():
        rows = parse(log)
        counts = Counter(row["match"] for row in rows)
        variants[name] = {
            "provenance": {
                "log": str(log), "log_sha256": file_hash(log),
                "script": str(script), "script_sha256": file_hash(script),
                "precision_caveat": "Percentages were printed to two decimals; recomputed metrics inherit that rounding.",
            },
            "num_parsed_rows": len(rows),
            "num_unique_matches": len(counts),
            "num_duplicate_match_ids": sum(1 for value in counts.values() if value > 1),
            "first_200": summarize(rows[:200]),
            "all_rows": summarize(rows),
            "rows": rows,
        }
    result = {
        "experiment_id": "E05_historical_audit",
        "metric_definitions": {
            "raw_possession_gain": "With-intervention simulated home possession minus no-intervention simulated home possession.",
            "absolute_error_reduction": "Absolute error of no-intervention versus real possession minus absolute error of with-intervention versus real possession.",
        },
        "paper_claims": {
            "random": {"mean_gain_pp": 1.02, "pct_improved": 32.55, "mean_gain_when_improved_pp": 3.12},
            "classic_markov": {"mean_gain_pp": 1.66, "pct_improved": 41.2, "mean_gain_when_improved_pp": 4.03},
            "plain_mcts": {"mean_gain_pp": 3.10, "pct_improved": 52.0, "mean_gain_when_improved_pp": 6.0},
            "priors_mcts": {"mean_gain_pp": 4.12, "pct_improved": 61.0, "mean_gain_when_improved_pp": 9.43},
        },
        "conclusion": "These surviving logs do not reproduce the Table 2 values under either plausible gain definition. Their sample construction also differs across variants, so they are audit evidence, not an accepted controlled comparison.",
        "variants": variants,
    }
    save_json(result, args.out_json)
    for name, variant in variants.items():
        print(name, variant["num_parsed_rows"], variant["num_unique_matches"], variant["first_200"])


if __name__ == "__main__":
    main()
