"""Run a command pinned to one GPU with a hard memory ceiling.

Polls `nvidia-smi` for the target GPU while the child process runs. If memory
used on that GPU exceeds --mem-limit-mib, the child is terminated (SIGTERM,
then SIGKILL after a grace period) before it can push the shared server into
an OOM/eviction situation.

Supports pinning to multiple GPUs (for DataParallel) via repeated --gpu flags;
every listed GPU is polled each interval and the child is aborted if ANY of
them exceeds --mem-limit-mib.

Usage:
    python3 -m impactful_possession_pipeline.run_guarded --gpu 0 \
        --mem-limit-mib 9000 --log impactful_possession_pipeline/runs/foo.memlog \
        -- python3 -m impactful_possession_pipeline.train_forward_simulator ...

    python3 -m impactful_possession_pipeline.run_guarded --gpu 0 --gpu 1 \
        --mem-limit-mib 9000 --log impactful_possession_pipeline/runs/foo.memlog \
        -- python3 -m impactful_possession_pipeline.train_forward_simulator ...
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def gpu_memory_used_mib(gpu_index: int) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, action="append", required=True, help="CUDA device index to pin the child process to; repeat for multiple GPUs (DataParallel).")
    parser.add_argument("--mem-limit-mib", type=int, default=9500, help="Abort child if GPU memory used on ANY pinned GPU exceeds this (of 11264 MiB total on these cards).")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--grace-seconds", type=float, default=10.0, help="Time between SIGTERM and SIGKILL on abort.")
    parser.add_argument("--log", required=True, help="Path to append memory-poll log lines to.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, after --")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("no command given after --", file=sys.stderr)
        sys.exit(2)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    gpus = args.gpu
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

    with open(log_path, "a", buffering=1) as log_f:
        def logline(msg: str) -> None:
            log_f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

        logline(f"launching on gpus={gpus} mem_limit_mib={args.mem_limit_mib} cmd={' '.join(command)}")
        proc = subprocess.Popen(command, env=env)

        aborted = False
        try:
            while True:
                ret = proc.poll()
                if ret is not None:
                    logline(f"child exited with code={ret}")
                    break
                used_by_gpu = {}
                poll_failed = False
                for g in gpus:
                    try:
                        used_by_gpu[g] = gpu_memory_used_mib(g)
                    except Exception as exc:
                        logline(f"nvidia-smi poll failed for gpu={g}: {exc}")
                        poll_failed = True
                if poll_failed:
                    time.sleep(args.poll_seconds)
                    continue
                logline(f"mem_used_mib={used_by_gpu}")
                over_limit = {g: u for g, u in used_by_gpu.items() if u > args.mem_limit_mib}
                if over_limit:
                    aborted = True
                    logline(f"ABORT: {over_limit} exceeded limit={args.mem_limit_mib}, sending SIGTERM")
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=args.grace_seconds)
                    except subprocess.TimeoutExpired:
                        logline("child did not exit after SIGTERM, sending SIGKILL")
                        proc.kill()
                        proc.wait()
                    break
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            logline("KeyboardInterrupt, terminating child")
            proc.terminate()
            proc.wait()
            raise

    if aborted:
        sys.exit(137)
    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    main()
