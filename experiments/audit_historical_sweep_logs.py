"""Inventory ambiguous historical sweep logs without overclaiming mappings."""

import hashlib
import re
from collections import Counter
from pathlib import Path

from impactful_possession_pipeline.utils import save_json
from paper_reproduction.parse_historical_policies import parse, summarize


MINUTE_RE = re.compile(r"^\s+Minute (\d+):")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minute_pattern(path: Path) -> list[int]:
    minutes = []
    in_first = False
    for line in path.read_text(errors="replace").splitlines():
        if "Processing " in line:
            if in_first:
                break
            in_first = True
            continue
        if in_first:
            match = MINUTE_RE.match(line)
            if match:
                minutes.append(int(match.group(1)))
    return minutes


def main() -> None:
    names = [
        "priors_1.txt", "priors_1_minute_sample.txt", "priors_3_250.txt",
        "priors_5.txt", "priors_7_250.txt", "priors_minute_sample_250.txt",
        "priors_3_500.txt", "priors_real_3_125.txt",
    ]
    logs = {}
    for name in names:
        path = Path("gpt_medium") / name
        rows = parse(path) if path.stat().st_size else []
        counts = Counter(row["match"] for row in rows)
        logs[name] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
            "num_rows": len(rows), "num_unique_matches": len(counts),
            "first_match_decision_minutes": minute_pattern(path),
            "first_200": summarize(rows[:200]) if rows else None,
        }
    result = {
        "scope": ["E06 intervention interval", "E07 search budget", "E08 lookahead horizon"],
        "logs": logs,
        "mapping_assessment": {
            "E06": "The two '*minute_sample*' logs visibly make decisions at every minute; ordinary logs use five-minute steps. No clearly identified ten-minute historical log exists.",
            "E07": "Filenames suggest priors search counts 1, 3, 5, and 7, but the paper requires 1, 5, 10, 30, and 50. Current source was edited in place and cannot authenticate every historical setting.",
            "E08": "Filenames expose 125, 250, and 500-like suffixes, while the paper requires 0, 40, 200, and 400 tokens. No exact trustworthy mapping is available.",
        },
        "conclusion": "These logs are useful historical traces but cannot reproduce E06-E08 as claimed. Exact controlled sweeps with explicit configs and a common pinned holdout remain required.",
    }
    save_json(result, "paper_reproduction/runs/e06_e08_historical_audit/results.json")
    for name, row in logs.items():
        print(name, row["bytes"], row["num_rows"], row["first_match_decision_minutes"][:12])


if __name__ == "__main__":
    main()
