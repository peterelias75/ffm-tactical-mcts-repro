import subprocess
import time
from pathlib import Path


PYTHON = "/home/peter/miniconda3/envs/fb-env/bin/python"
BASE = Path("impactful_possession_pipeline/runs")
RUN_DIR = BASE / "forward_simulator_full_3400matches_transition_ft_5epochs_20260703_maxlen2048_retry"
CKPT = RUN_DIR / "best_forward_simulator.pt"
OUT_JSON = RUN_DIR / "rollout_100matches_transition_ft_lb30_cond005.json"
COND_PRIOR = BASE / "forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/rollout_conditional_priors_45_60_holdout.json"


def checkpoint_stable(path: Path) -> bool:
    if not path.exists():
        return False
    first = path.stat().st_size
    time.sleep(20)
    return path.exists() and path.stat().st_size == first and first > 0


def main() -> None:
    while not checkpoint_stable(CKPT):
        time.sleep(60)

    args = [
        PYTHON,
        "-u",
        "-m",
        "impactful_possession_pipeline.evaluate_rollout_diagnostics",
        "--checkpoint",
        str(CKPT),
        "--out-json",
        str(OUT_JSON),
        "--max-files",
        "100",
        "--cut-minute",
        "45",
        "--end-minute",
        "60",
        "--device",
        "cuda:0",
        "--density-from-prefix",
        "--density-lookback-minutes",
        "30",
        "--event-density-tolerance",
        "2",
        "--prefix-possession-target",
        "--possession-lookback-minutes",
        "30",
        "--external-possession-weight",
        "0.75",
        "--possession-guidance-strength",
        "16",
        "--conditional-prior-json",
        str(COND_PRIOR),
        "--conditional-prior-bias-strength",
        "0.05",
    ]
    with (RUN_DIR / "post_train_benchmark.log").open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(args)}\n")
        log.flush()
        proc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nexit_code={proc.returncode}\n")


if __name__ == "__main__":
    main()
