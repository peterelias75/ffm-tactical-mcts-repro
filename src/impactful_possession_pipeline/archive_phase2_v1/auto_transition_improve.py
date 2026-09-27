import json
import subprocess
import time
from pathlib import Path


PYTHON = "/home/peter/miniconda3/envs/fb-env/bin/python"
BASE = Path("impactful_possession_pipeline/runs")
SOURCE_CKPT = BASE / "forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/best_forward_simulator.pt"
COND_PRIOR = BASE / "forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/rollout_conditional_priors_45_60_holdout.json"
TARGET_MAE = 5.0
BENCHMARK_TO_BEAT = 6.816347882892968


def wait_for_json(path: Path) -> dict:
    while True:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(60)


def run(args: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(args)}\n")
        log.flush()
        proc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nexit_code={proc.returncode}\n")
        return int(proc.returncode)


def train(run_dir: Path, resume_ckpt: Path, epochs: int, lr: str, possession_weight: str, team_weight: str, pflip_weight: str, event_weight: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"{run_dir}_training.log")
    args = [
        PYTHON,
        "-u",
        "-m",
        "impactful_possession_pipeline.train_forward_simulator",
        "--events-dir",
        "/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/events",
        "--lineups-dir",
        "/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/lineups",
        "--out-dir",
        str(run_dir),
        "--max-matches",
        "3400",
        "--epochs",
        str(epochs),
        "--batch-size",
        "4",
        "--num-workers",
        "2",
        "--max-len",
        "2048",
        "--resume-checkpoint",
        str(resume_ckpt),
        "--reset-best-metric",
        "--lr",
        lr,
        "--possession-weight",
        possession_weight,
        "--count-weight",
        "0.15",
        "--uncertainty-weight",
        "0.05",
        "--event-token-weight",
        event_weight,
        "--team-token-weight",
        team_weight,
        "--possession-flip-token-weight",
        pflip_weight,
        "--role-token-weight",
        "1.0",
        "--location-token-weight",
        "1.0",
        "--goal-token-weight",
        "1.0",
    ]
    run(args, log_path)


def benchmark(run_dir: Path) -> float:
    ckpt = run_dir / "best_forward_simulator.pt"
    out_json = run_dir / "rollout_100matches_transition_ft_lb30_cond005.json"
    log_path = run_dir / "auto_benchmark.log"
    args = [
        PYTHON,
        "-u",
        "-m",
        "impactful_possession_pipeline.evaluate_rollout_diagnostics",
        "--checkpoint",
        str(ckpt),
        "--out-json",
        str(out_json),
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
    run(args, log_path)
    data = wait_for_json(out_json)
    return float(data["summary"]["mean_abs_possession_error_pp"])


def main() -> None:
    current_dir = BASE / "forward_simulator_full_3400matches_transition_ft_5epochs_20260703_maxlen2048"
    current_benchmark = current_dir / "rollout_100matches_transition_ft_lb30_cond005.json"
    status_path = BASE / "transition_auto_improve_status.json"

    data = wait_for_json(current_benchmark)
    first_mae = float(data["summary"]["mean_abs_possession_error_pp"])
    status = {"first_run": str(current_dir), "first_mae_pp": first_mae, "target_mae_pp": TARGET_MAE}
    status_path.write_text(json.dumps(status, indent=2))
    if first_mae <= TARGET_MAE:
        status["status"] = "target_met"
        status_path.write_text(json.dumps(status, indent=2))
        return

    if first_mae < BENCHMARK_TO_BEAT:
        next_dir = BASE / "forward_simulator_full_3400matches_transition_ft_extend_3epochs_20260703_maxlen2048"
        train(next_dir, current_dir / "best_forward_simulator.pt", 3, "0.00003", "8.0", "3.0", "6.0", "1.5")
    else:
        next_dir = BASE / "forward_simulator_full_3400matches_transition_ft_mild_4epochs_20260703_maxlen2048"
        train(next_dir, SOURCE_CKPT, 4, "0.00003", "6.0", "2.0", "3.0", "1.25")

    second_mae = benchmark(next_dir)
    status.update({"second_run": str(next_dir), "second_mae_pp": second_mae, "status": "target_met" if second_mae <= TARGET_MAE else "needs_next_iteration"})
    status_path.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
