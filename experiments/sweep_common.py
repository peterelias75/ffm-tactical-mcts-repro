"""Shared helpers for the E06/E07/E08 sweep runners and the E09 aggregator.

Directory conventions mirror paper_reproduction/launch_two_gpu_validation.sh:
every sweep point ("condition") gets its own subdirectory under --run-root
containing:

  command.sh   -- the exact shell command that was (or would be) run
  stdout.log   -- appended stdout/stderr of every subprocess.run for this
                  condition (a resumed condition appends a new "=== sweep run
                  start ... ===" marker rather than overwriting)
  results.json -- the underlying evaluate_mcts_pipeline.py --out-json output
                  (has its own "summary" + "matches" schema and its own
                  internal --resume/per-match resumability)
  status       -- "running" | "complete" | "failed:<returncode>"
  timing.json  -- wall-clock elapsed seconds for the subprocess call plus
                  before/after match counts (used for E07's latency-per-
                  decision estimate; harmless bookkeeping for E06/E08)

A condition is skipped (no subprocess launched) if its results.json already
reports complete=True with num_matches >= the number of matches requested for
this invocation -- this is what makes "run once as a --max-matches 2 smoke
test, then re-invoke with a larger --max-matches on the same --run-root"
safe: already-computed matches are never redone, and evaluate_mcts_pipeline.py's
own --out-json/--resume mechanism means even a resumed subprocess call only
recomputes the matches that are actually new.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/home/peter/fb-proj")
PYTHON = "/home/peter/miniconda3/envs/fb-env/bin/python"
DEFAULT_CHECKPOINT = str(
    REPO_ROOT / "impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt"
)
DEFAULT_HOLDOUT = str(REPO_ROOT / "paper_reproduction/holdout_table2_200.json")
DEFAULT_SEED = 20260727


def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path, data) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def holdout_match_count(holdout_json: str, max_matches: Optional[int]) -> int:
    holdout = load_json(holdout_json)
    n = len(holdout["match_ids"])
    return min(n, max_matches) if max_matches else n


def decisions_per_match(step_minutes: int, cut_minute: int = 45, end_minute: int = 90) -> int:
    return len(range(cut_minute, end_minute, step_minutes))


def condition_already_complete(out_json: Path, expected_matches: int) -> bool:
    if not out_json.exists():
        return False
    try:
        data = load_json(out_json)
    except Exception:
        return False
    summary = data.get("summary", {})
    return bool(summary.get("complete")) and summary.get("num_matches", 0) >= expected_matches


def run_condition(
    cond_dir: Path,
    module: str,
    cli_args: list,
    gpu: int,
    out_json: Path,
    expected_matches: int,
    label: str,
) -> dict:
    """Run (or skip, if already complete) one sweep point.

    Returns a dict with: skipped, elapsed_seconds, returncode, matches_before,
    matches_after, newly_processed, summary (the out_json "summary" dict, or
    {} if the run failed before producing one).
    """
    if gpu not in (0, 1, 2):
        raise ValueError(f"physical GPU {gpu} is outside the permitted set {{0, 1, 2}}")
    cond_dir.mkdir(parents=True, exist_ok=True)
    status_path = cond_dir / "status"
    command_path = cond_dir / "command.sh"
    stdout_path = cond_dir / "stdout.log"
    timing_path = cond_dir / "timing.json"
    config_path = cond_dir / "config.json"
    gpu_log_path = cond_dir / "gpu_memory.csv"

    full_cmd = [PYTHON, "-u", "-m", module] + [str(a) for a in cli_args]
    shell_cmd = "env CUDA_VISIBLE_DEVICES=%d %s" % (gpu, shlex.join(full_cmd))
    command_path.write_text(shell_cmd + "\n")
    save_json(config_path, {
        "label": label,
        "module": module,
        "cli_args": [str(a) for a in cli_args],
        "physical_gpu": gpu,
        "cuda_process_memory_fraction": 0.84,
        "memory_warning_mib": 9500,
        "expected_matches": expected_matches,
        "out_json": str(out_json),
    })

    matches_before = 0
    if out_json.exists():
        try:
            matches_before = load_json(out_json).get("summary", {}).get("num_matches", 0)
        except Exception:
            matches_before = 0

    if condition_already_complete(out_json, expected_matches):
        status_path.write_text("complete\n")
        summary = load_json(out_json)["summary"]
        timing = load_json(timing_path) if timing_path.exists() else {}
        print(f"[{label}] already complete ({matches_before} matches) -- skipping", flush=True)
        return {
            "skipped": True,
            "elapsed_seconds": timing.get("elapsed_seconds"),
            "returncode": 0,
            "matches_before": timing.get("matches_before", matches_before),
            "matches_after": timing.get("matches_after", matches_before),
            "newly_processed": timing.get("newly_processed", 0),
            "summary": summary,
        }

    status_path.write_text("running\n")
    print(f"[{label}] launching: {shell_cmd}", flush=True)
    t0 = time.perf_counter()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    monitor = subprocess.Popen([
        "/bin/bash", str(REPO_ROOT / "paper_reproduction/monitor_gpu_memory.sh"),
        str(gpu), str(gpu_log_path), "9500",
    ], cwd=str(REPO_ROOT))
    try:
        with open(stdout_path, "a") as log_f:
            log_f.write(f"\n=== sweep run start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
            log_f.flush()
            proc = subprocess.run(
                full_cmd, cwd=str(REPO_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT,
            )
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
    elapsed = time.perf_counter() - t0

    matches_after = matches_before
    summary = {}
    if out_json.exists():
        try:
            summary = load_json(out_json)["summary"]
            matches_after = summary.get("num_matches", matches_before)
        except Exception:
            pass

    ok = proc.returncode == 0 and bool(summary.get("complete"))
    status_path.write_text(("complete\n" if ok else f"failed:{proc.returncode}\n"))

    timing = {
        "elapsed_seconds": elapsed,
        "matches_before": matches_before,
        "matches_after": matches_after,
        "newly_processed": max(0, matches_after - matches_before),
    }
    save_json(timing_path, timing)

    print(f"[{label}] done in {elapsed:.1f}s returncode={proc.returncode} matches={matches_after}", flush=True)
    return {
        "skipped": False,
        "elapsed_seconds": elapsed,
        "returncode": proc.returncode,
        "matches_before": matches_before,
        "matches_after": matches_after,
        "newly_processed": timing["newly_processed"],
        "summary": summary,
    }
