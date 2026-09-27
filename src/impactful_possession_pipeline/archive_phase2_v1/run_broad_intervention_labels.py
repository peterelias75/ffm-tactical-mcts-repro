import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .utils import mean, save_json


RUN_DIR = Path("impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048")
CHECKPOINT = RUN_DIR / "best_forward_simulator.pt"
VALUE_HEAD_JSON = RUN_DIR / "value_head_interventions_final_holdout.json"
OUT_DIR = RUN_DIR / "calibration_labels_vh015_s1"
EVENTS_DIR = Path("/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/events")

JOBS = [
    ("sub_attack", "ACTION::HOME_SUB_ATTACK", "cuda:0"),
    ("sub_neutral", "ACTION::HOME_SUB_NEUTRAL", "cuda:1"),
    ("tactic_attack", "ACTION::HOME_TACTIC_ATTACK", "cuda:2"),
    ("tactic_neutral", "ACTION::HOME_TACTIC_NEUTRAL", "cuda:3"),
]


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def data_mount_ready() -> bool:
    cmd = ["timeout", "10", "find", str(EVENTS_DIR), "-maxdepth", "1", "-name", "*.json", "-print", "-quit"]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
    except Exception as exc:
        log(f"data mount probe failed: {exc}")
        return False
    if result.returncode == 0 and result.stdout.strip():
        return True
    detail = (result.stderr or result.stdout or "not ready").strip().splitlines()
    log(f"data mount not ready: {detail[-1] if detail else result.returncode}")
    return False

def wait_for_data_mount(poll_seconds: int) -> None:
    attempt = 0
    while not data_mount_ready():
        attempt += 1
        log(f"waiting for data mount attempt {attempt}; next check in {poll_seconds}s")
        time.sleep(poll_seconds)
    log(f"data mount is ready: {EVENTS_DIR}")


def command_for_job(name: str, action: str, device: str, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "impactful_possession_pipeline.evaluate_shortlisted_interventions",
        "--checkpoint",
        str(args.checkpoint),
        "--value-head-json",
        str(args.value_head_json),
        "--out-json",
        str(args.out_dir / f"{name}.json"),
        "--score-action",
        action,
        "--min-value-head-delta",
        str(args.min_value_head_delta),
        "--rollout-seeds",
        str(args.rollout_seeds),
        "--seed",
        str(args.seed),
        "--device",
        device,
        "--cut-minute",
        str(args.cut_minute),
        "--end-minute",
        str(args.end_minute),
        "--prompt-tail-tokens",
        "512",
        "--possession-guidance-strength",
        "12",
        "--density-from-prefix",
        "--density-lookback-minutes",
        "15",
        "--density-multiplier",
        "1.12",
        "--event-density-tolerance",
        "2",
        "--prefix-possession-target",
        "--possession-lookback-minutes",
        "15",
        "--external-possession-weight",
        "0.75",
        "--enforce-possession-state",
        "--football-structure-constraints",
        "--pass-receipt-boost",
        "1.0",
        "--pass-carry-boost",
        "0.4",
        "--overproduced-event-penalty",
        "0.5",
        "--max-pass-run",
        "4",
        "--force-receipt-after-pass-run",
        "3",
    ]


def run_jobs(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for name, action, device in JOBS:
        out_json = args.out_dir / f"{name}.json"
        if out_json.exists() and not args.overwrite:
            log(f"skipping existing {out_json}")
            continue
        log_path = args.out_dir / f"{name}.log"
        log(f"starting {name} on {device}: {action}")
        handle = log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(
            command_for_job(name, action, device, args),
            cwd=Path.cwd(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((name, proc, handle, log_path))

    exit_code = 0
    for name, proc, handle, log_path in processes:
        code = proc.wait()
        handle.close()
        log(f"{name} finished with code {code}; log={log_path}")
        if code != 0:
            exit_code = code
    return exit_code


def aggregate(out_dir: Path) -> dict:
    rows = []
    action_summaries = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {"action_calibration_summary.json", "aggregate_vh015_s1.json"}:
            continue
        data = json.loads(path.read_text())
        action = data.get("summary", {}).get("score_action") or path.stem
        action_rows = data.get("matches", [])
        rows.extend(action_rows)
        action_summaries[action] = {
            "file": path.name,
            "n": len(action_rows),
            "mean_value_head_delta_pp": mean([row.get("value_head_delta_pp", 0.0) for row in action_rows]),
            "mean_rollout_delta_pp": mean([row.get("mean_rollout_delta_pp", 0.0) for row in action_rows]),
            "positive_rate": mean([1.0 if row.get("mean_rollout_delta_pp", 0.0) > 0.0 else 0.0 for row in action_rows]),
            "mean_positive_seed_rate": mean([row.get("positive_seed_rate", 0.0) for row in action_rows]),
        }

    summary = {
        "num_actions": len(action_summaries),
        "num_rows": len(rows),
        "mean_value_head_delta_pp": mean([row.get("value_head_delta_pp", 0.0) for row in rows]),
        "mean_rollout_delta_pp": mean([row.get("mean_rollout_delta_pp", 0.0) for row in rows]),
        "positive_rate": mean([1.0 if row.get("mean_rollout_delta_pp", 0.0) > 0.0 else 0.0 for row in rows]),
        "by_action": action_summaries,
    }
    output = {"summary": summary, "matches": rows}
    save_json(output, str(out_dir / "aggregate_vh015_s1.json"))
    save_json(action_summaries, str(out_dir / "action_calibration_summary.json"))
    return output


def rerun_scorer(args: argparse.Namespace) -> int:
    log("rerunning offline intervention scorer with new labels")
    return subprocess.call(
        [sys.executable, "-m", "impactful_possession_pipeline.train_intervention_scorer", "--out-json", str(args.scorer_out_json)],
        cwd=Path.cwd(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for data and run broad paired intervention labels.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--value-head-json", type=Path, default=VALUE_HEAD_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scorer-out-json", type=Path, default=Path("impactful_possession_pipeline/runs/intervention_scorer_summary_vh015_s1.json"))
    parser.add_argument("--min-value-head-delta", type=float, default=0.015)
    parser.add_argument("--rollout-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--end-minute", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    if args.no_wait and not data_mount_ready():
        raise RuntimeError(f"data mount is not ready: {EVENTS_DIR}")
    if not args.no_wait:
        wait_for_data_mount(args.poll_seconds)

    code = run_jobs(args)
    aggregate_output = aggregate(args.out_dir)
    aggregate_path = args.out_dir / "aggregate_vh015_s1.json"
    log(f"saved aggregate: {aggregate_path}")
    log(json.dumps(aggregate_output["summary"], indent=2))
    scorer_code = rerun_scorer(args)
    raise SystemExit(code or scorer_code)


if __name__ == "__main__":
    main()
