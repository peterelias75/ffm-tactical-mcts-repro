import json
import subprocess
import time
from pathlib import Path


PYTHON = "/home/peter/miniconda3/envs/fb-env/bin/python"
BASE = Path("impactful_possession_pipeline/runs")

VALIDATIONS = {
    "seed7_100_45_60": BASE / "posw6_best_seed_check_20260703/rollout_best_den1125_100matches_seed7.json",
    "seed123_100_45_60": BASE / "posw6_best_seed_check_20260703/rollout_best_den1125_100matches_seed123.json",
    "seed42_340_45_60": BASE / "posw6_best_validation_20260703/rollout_best_den1125_340matches_seed42.json",
    "seed42_100_30_45": BASE / "posw6_best_window_tests_20260703/rollout_best_den1125_100matches_window30_45_seed42.json",
}

SOURCE_CKPT = BASE / "forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/best_forward_simulator.pt"
TRANSITION_DIR = BASE / "forward_simulator_full_3400matches_transition_ft_5epochs_20260703_maxlen2048_retry"
COND_PRIOR = BASE / "forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048/rollout_conditional_priors_45_60_holdout.json"
SUMMARY_PATH = BASE / "best_validation_then_train_status.json"


def read_json_when_ready(path: Path) -> dict:
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


def summarize_validations() -> dict:
    results = {}
    for name, path in VALIDATIONS.items():
        data = read_json_when_ready(path)
        summary = data.get("summary", {})
        results[name] = {
            "path": str(path),
            "mean_abs_possession_error_pp": summary.get("mean_abs_possession_error_pp"),
            "team_share_l1": summary.get("team_share_l1"),
            "event_type_l1": summary.get("event_type_l1"),
            "illegal_possession_switch_rate": summary.get("illegal_possession_switch_rate"),
        }
        SUMMARY_PATH.write_text(json.dumps({"status": "waiting_or_summarizing", "results": results}, indent=2))
    return results


def train_transition() -> None:
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
        str(TRANSITION_DIR),
        "--max-matches",
        "3400",
        "--epochs",
        "5",
        "--batch-size",
        "4",
        "--num-workers",
        "2",
        "--max-len",
        "2048",
        "--resume-checkpoint",
        str(SOURCE_CKPT),
        "--reset-best-metric",
        "--lr",
        "0.00005",
        "--possession-weight",
        "8.0",
        "--count-weight",
        "0.15",
        "--uncertainty-weight",
        "0.05",
        "--event-token-weight",
        "1.5",
        "--team-token-weight",
        "3.0",
        "--possession-flip-token-weight",
        "6.0",
        "--role-token-weight",
        "1.0",
        "--location-token-weight",
        "1.0",
        "--goal-token-weight",
        "1.0",
    ]
    run(args, Path(f"{TRANSITION_DIR}_training.log"))


def benchmark_transition() -> None:
    ckpt = TRANSITION_DIR / "best_forward_simulator.pt"
    out_json = TRANSITION_DIR / "rollout_100matches_transition_ft_lb30_cond005.json"
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
    run(args, TRANSITION_DIR / "post_train_benchmark.log")


def main() -> None:
    results = summarize_validations()
    best_mae = min(
        float(v["mean_abs_possession_error_pp"])
        for v in results.values()
        if v["mean_abs_possession_error_pp"] is not None
    )
    SUMMARY_PATH.write_text(json.dumps({"status": "validation_done_training_transition", "best_mae_pp": best_mae, "results": results}, indent=2))
    train_transition()
    benchmark_transition()
    SUMMARY_PATH.write_text(json.dumps({"status": "transition_retry_done", "best_mae_pp": best_mae, "results": results, "transition_run": str(TRANSITION_DIR)}, indent=2))


if __name__ == "__main__":
    main()
