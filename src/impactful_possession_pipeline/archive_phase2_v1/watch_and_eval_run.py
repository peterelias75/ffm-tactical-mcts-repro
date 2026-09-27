import argparse
import subprocess
import time
from pathlib import Path


def process_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def run_command(args: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(args)}\n")
        log.flush()
        proc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nexit_code={proc.returncode}\n")
        return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a training PID to finish, then run checkpoint evaluation.")
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python", default="/home/peter/miniconda3/envs/fb-env/bin/python")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--rollout-max-files", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_log = run_dir / "post_train_eval.log"
    checkpoint = Path(args.checkpoint)

    with eval_log.open("a", encoding="utf-8") as log:
        log.write(f"watching train_pid={args.train_pid}\n")
        log.flush()

    while process_alive(args.train_pid):
        time.sleep(args.poll_seconds)

    with eval_log.open("a", encoding="utf-8") as log:
        log.write("training process exited; waiting for checkpoint file\n")
        log.flush()

    while not checkpoint.exists():
        time.sleep(args.poll_seconds)

    heads_json = run_dir / "holdout_heads.json"
    rollout_json = run_dir / "rollout_diagnostics.json"

    run_command(
        [
            args.python,
            "-m",
            "impactful_possession_pipeline.evaluate_holdout_heads",
            "--checkpoint",
            str(checkpoint),
            "--out-json",
            str(heads_json),
        ],
        eval_log,
    )
    run_command(
        [
            args.python,
            "-m",
            "impactful_possession_pipeline.evaluate_rollout_diagnostics",
            "--checkpoint",
            str(checkpoint),
            "--max-files",
            str(args.rollout_max_files),
            "--cut-minute",
            "45",
            "--end-minute",
            "60",
            "--out-json",
            str(rollout_json),
        ],
        eval_log,
    )


if __name__ == "__main__":
    main()
