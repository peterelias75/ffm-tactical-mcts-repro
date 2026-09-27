"""Watch a small risk-gated MCTS run and auto-scale if it clears gates."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-json', required=True)
    parser.add_argument('--source-holdout-json', default='impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json')
    parser.add_argument('--next-holdout-json', default='impactful_possession_pipeline/runs/mcts_risk_20.json')
    parser.add_argument('--next-out-json', default='impactful_possession_pipeline/runs/full_holdout_checks/mcts_risk_20.json')
    parser.add_argument('--next-log', default='impactful_possession_pipeline/runs/full_holdout_checks/mcts_risk_20_stdout.log')
    parser.add_argument('--next-memlog', default='impactful_possession_pipeline/runs/full_holdout_checks/mcts_risk_20.memlog')
    parser.add_argument('--report-json', default='impactful_possession_pipeline/runs/full_holdout_checks/mcts_risk_autoscale_report.json')
    parser.add_argument('--screen-name', default='mcts_risk_20')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--mem-limit-mib', type=int, default=5000)
    parser.add_argument('--num-matches', type=int, default=20)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--min-mean-pp', type=float, default=0.0)
    parser.add_argument('--min-positive-rate', type=float, default=50.0)
    parser.add_argument('--max-std-pp', type=float, default=12.0)
    args = parser.parse_args()

    watch_path = Path(args.watch_json)
    report_path = Path(args.report_json)
    report = {'status': 'watching', 'watch_json': str(watch_path), 'launched': False}
    save_json(report_path, report)

    while True:
        if watch_path.exists():
            try:
                watched = load_json(watch_path)
                summary = watched.get('summary', {})
                if summary.get('complete'):
                    break
            except Exception as exc:
                report['last_error'] = str(exc)
                save_json(report_path, report)
        time.sleep(args.poll_seconds)

    summary = load_json(watch_path).get('summary', {})
    gates = {
        'mean_delta_pp': summary.get('mean_delta_pp', 0.0) >= args.min_mean_pp,
        'pct_matches_improved': summary.get('pct_matches_improved', 0.0) >= args.min_positive_rate,
        'std_delta_pp': summary.get('std_delta_pp', 999.0) <= args.max_std_pp,
    }
    should_launch = all(gates.values())
    report = {
        'status': 'gated',
        'watch_json': str(watch_path),
        'summary': summary,
        'gates': gates,
        'launched': False,
    }

    if should_launch:
        source = load_json(Path(args.source_holdout_json))
        match_ids = source['match_ids'][: args.num_matches]
        save_json(Path(args.next_holdout_json), {'match_ids': match_ids})
        command = [
            'screen', '-dmS', args.screen_name,
            'bash', '-lc',
            'cd /home/peter/fb-proj; '
            f'/home/peter/miniconda3/envs/fb-env/bin/python3 -m impactful_possession_pipeline.run_guarded '
            f'--gpu {args.gpu} --mem-limit-mib {args.mem_limit_mib} --poll-seconds 10 '
            f'--log {args.next_memlog} -- '
            f'/home/peter/miniconda3/envs/fb-env/bin/python3 -u -m impactful_possession_pipeline.evaluate_mcts_pipeline '
            f'--checkpoint impactful_possession_pipeline/runs/phase1_v3_richactions_24epochs_20260722/epoch_checkpoints/epoch016.pt '
            f'--holdout-json {args.next_holdout_json} --num-simulations 5 --c-puct 6.0 '
            f'--selection-policy risk_adjusted --risk-penalty 1.0 --noop-margin 0.01 '
            f'--out-json {args.next_out_json} --resume > {args.next_log} 2>&1'
        ]
        subprocess.run(command, check=True)
        report.update({
            'status': 'launched_next',
            'launched': True,
            'next_holdout_json': args.next_holdout_json,
            'next_out_json': args.next_out_json,
            'screen_name': args.screen_name,
            'gpu': args.gpu,
            'mem_limit_mib': args.mem_limit_mib,
        })
    else:
        report['status'] = 'stopped_by_gates'

    save_json(report_path, report)


if __name__ == '__main__':
    main()
