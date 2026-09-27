"""Parse the complete historical 600 x 10 stability log into JSON evidence."""

import argparse
import hashlib
import re
from pathlib import Path

from impactful_possession_pipeline.utils import mean, save_json


RUN_RE = re.compile(r"^File (.+\.json), Run (\d+): Simulated Home Possession = ([0-9.]+)%$")
MATCH_RE = re.compile(r"^Match (.+) Summary:$")
FT_RE = re.compile(r"^  FT Home Possession: ([0-9.]+)%$")
MEAN_RE = re.compile(r"^  Simulated Home Possession \(Mean over (\d+) runs\): ([0-9.]+)%$")
VAR_RE = re.compile(r"^  Variance: ([0-9.]+)$")
DIFF_RE = re.compile(r"^  Absolute Difference: ([0-9.]+)%$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="model_v3/model_v3_mean_var.txt")
    parser.add_argument("--script", default="model_v3/model_v3_mean_var.py")
    parser.add_argument("--out-json", default="paper_reproduction/runs/e04_repeat_stability/results.json")
    args = parser.parse_args()

    log_path, script_path = Path(args.log), Path(args.script)
    lines = log_path.read_text(errors="replace").splitlines()
    simulations: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for index, line in enumerate(lines):
        match = RUN_RE.match(line)
        if match:
            name, run_index, value = match.groups()
            simulations.setdefault(name.removesuffix(".json"), []).append({
                "run_index": int(run_index), "simulated_home_possession_pp": float(value)
            })
            continue
        match = MATCH_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        fields = lines[index + 1:index + 5]
        ft, avg, variance, diff = FT_RE.match(fields[0]), MEAN_RE.match(fields[1]), VAR_RE.match(fields[2]), DIFF_RE.match(fields[3])
        if not all((ft, avg, variance, diff)):
            raise RuntimeError(f"malformed summary near line {index + 1}")
        summaries[name] = {
            "match": name + ".json",
            "real_home_possession_pp": float(ft.group(1)),
            "runs_per_match": int(avg.group(1)),
            "mean_simulated_home_possession_pp": float(avg.group(2)),
            "sample_variance_pp2": float(variance.group(1)),
            "absolute_error_pp": float(diff.group(1)),
            "simulations": simulations.get(name, []),
        }

    rows = list(summaries.values())
    bad_runs = [row["match"] for row in rows if len(row["simulations"]) != row["runs_per_match"]]
    duplicate_indices = [
        row["match"] for row in rows
        if sorted(item["run_index"] for item in row["simulations"]) != list(range(1, row["runs_per_match"] + 1))
    ]
    if len(rows) != 600 or bad_runs or duplicate_indices:
        raise RuntimeError(f"incomplete log: matches={len(rows)} bad_runs={bad_runs[:5]} bad_indices={duplicate_indices[:5]}")

    result = {
        "experiment_id": "E04",
        "provenance": {
            "type": "parsed_complete_historical_stdout",
            "log": str(log_path), "log_sha256": sha256(log_path),
            "script": str(script_path), "script_sha256": sha256(script_path),
            "model_path_declared_by_script": "model_v3/trained_transformer_model_v3.pt",
            "model_path_currently_present": False,
            "warning": "The complete output log is preserved, but the script-declared model filename is not currently present. This proves the historical run/result, but prevents a bit-for-bit clean rerun until the original weight file is identified.",
        },
        "metric_definition": "The historical script uses statistics.variance, i.e. sample variance (N-1 denominator), over ten simulated home-possession percentages per match.",
        "summary": {
            "num_matches": len(rows),
            "runs_per_match": 10,
            "num_individual_simulations": sum(len(row["simulations"]) for row in rows),
            "mean_within_match_sample_variance_pp2": mean([row["sample_variance_pp2"] for row in rows]),
            "mean_absolute_error_pp": mean([row["absolute_error_pp"] for row in rows]),
            "paper_variance_claim_pp2": 13.87,
            "paper_error_claim_pp": 12.02,
            "complete": True,
        },
        "matches": rows,
    }
    save_json(result, args.out_json)
    print(result["summary"])


if __name__ == "__main__":
    main()
