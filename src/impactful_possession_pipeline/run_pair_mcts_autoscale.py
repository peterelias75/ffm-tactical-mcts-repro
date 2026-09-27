# Watch paired-delta MCTS pilots and launch one guarded scale-up if gates pass.

import json
import subprocess
import time
from pathlib import Path

RUN_DIR = Path("impactful_possession_pipeline/runs/full_holdout_checks")
VARIANTS = [
    {"name": "m01", "json": RUN_DIR / "mcts_pair_m01.json", "noop_margin": 0.01},
    {"name": "m02", "json": RUN_DIR / "mcts_pair_m02.json", "noop_margin": 0.02},
]
REPORT = RUN_DIR / "mcts_pair_autoscale_report.json"
SOURCE_HOLDOUT = Path("impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json")
NEXT_HOLDOUT = Path("impactful_possession_pipeline/runs/mcts_pair_best20.json")
NEXT_OUT = RUN_DIR / "mcts_pair_best20.json"
NEXT_LOG = RUN_DIR / "mcts_pair_best20_stdout.log"
NEXT_MEMLOG = RUN_DIR / "mcts_pair_best20.memlog"
CHECKPOINT = "impactful_possession_pipeline/runs/phase1_v3_richactions_24epochs_20260722/epoch_checkpoints/epoch016.pt"


def load_json(path: Path):
    return json.loads(path.read_text())


def save_report(payload: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True))


def summary_for(variant: dict) -> dict:
    path = variant["json"]
    if not path.exists():
        return {"name": variant["name"], "path": str(path), "exists": False, "complete": False}
    data = load_json(path)
    summary = data.get("summary", {})
    return {"name": variant["name"], "path": str(path), "exists": True, **summary}


def passes_gate(summary: dict) -> bool:
    return (
        summary.get("complete") is True
        and summary.get("mean_delta_pp", -999.0) >= 1.0
        and summary.get("pct_matches_improved", 0.0) >= 50.0
        and summary.get("std_delta_pp", 999.0) <= 10.0
    )


def best_variant(summaries: list[dict]) -> dict | None:
    passing = [s for s in summaries if passes_gate(s)]
    if not passing:
        return None
    return max(
        passing,
        key=lambda s: (s.get("pct_matches_improved", 0.0), s.get("mean_delta_pp", -999.0), -s.get("std_delta_pp", 999.0)),
    )


def create_next_holdout() -> None:
    source = load_json(SOURCE_HOLDOUT)
    NEXT_HOLDOUT.write_text(json.dumps({"match_ids": source["match_ids"][:20]}, indent=2))


def launch_scaleup(best: dict) -> None:
    margin = 0.01 if best["name"] == "m01" else 0.02
    create_next_holdout()
    cmd = (
        "screen -dmS mcts_pair_best20 bash -lc "
        "'cd /home/peter/fb-proj; "
        "/home/peter/miniconda3/envs/fb-env/bin/python3 -m impactful_possession_pipeline.run_guarded "
        "--gpu 2 --mem-limit-mib 5000 --poll-seconds 10 "
        f"--log {NEXT_MEMLOG} -- "
        "/home/peter/miniconda3/envs/fb-env/bin/python3 -u -m impactful_possession_pipeline.evaluate_mcts_pipeline "
        f"--checkpoint {CHECKPOINT} "
        f"--holdout-json {NEXT_HOLDOUT} "
        "--num-simulations 5 --c-puct 6.0 --selection-policy risk_adjusted --risk-penalty 1.0 "
        f"--noop-margin {margin} --reward-mode paired_delta "
        f"--out-json {NEXT_OUT} --resume "
        f"> {NEXT_LOG} 2>&1'"
    )
    subprocess.check_call(cmd, shell=True)


def main() -> None:
    launched = NEXT_OUT.exists()
    while True:
        summaries = [summary_for(v) for v in VARIANTS]
        complete = all(s.get("complete") is True for s in summaries)
        report = {"status": "watching", "variants": summaries, "launched": launched}
        save_report(report)
        if complete:
            best = best_variant(summaries)
            if best is None:
                save_report({"status": "stopped_by_gates", "variants": summaries, "launched": False})
                return
            if not launched:
                launch_scaleup(best)
                launched = True
            save_report({"status": "launched_scaleup", "best": best, "variants": summaries, "next_out": str(NEXT_OUT), "launched": True})
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
