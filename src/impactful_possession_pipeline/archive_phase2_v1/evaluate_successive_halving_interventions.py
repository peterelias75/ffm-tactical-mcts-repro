import argparse
from pathlib import Path

import torch

from .config import DecodeConfig, TokenConfig
from .dataset import token_index_at_cut
from .evaluate_forced_timing_candidates_sanity import (
    make_decode_cfg,
    prefix_home_possession_target,
    replay_with_metrics,
    sanity_reasons,
    sem_lcb,
)
from .model import load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import home_possession_ratio, load_json, mean, save_json, std
from .vocabulary import Vocab


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(',') if part.strip()]


def select_candidates(data: dict, top_k: int) -> list[dict]:
    candidates = data.get('recommended_validation') or data.get('top_candidates') or data.get('candidates') or []
    return candidates[:top_k]


def summarize_rows(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row.get('accepted')]
    deltas = [float(row['delta_pp']) for row in accepted]
    all_deltas = [float(row['delta_pp']) for row in rows]
    downside = [max(0.0, -value) for value in deltas]
    return {
        'n': len(deltas),
        'all_n': len(all_deltas),
        'mean_delta_pp': mean(deltas),
        'std_delta_pp': std(deltas),
        'lcb90_delta_pp': sem_lcb(deltas),
        'positive_rate': mean([1.0 if value > 0.0 else 0.0 for value in deltas]),
        'min_delta_pp': min(deltas) if deltas else 0.0,
        'max_downside_pp': max(downside) if downside else 0.0,
        'mean_downside_pp': mean(downside),
        'all_mean_delta_pp': mean(all_deltas),
    }


def utility(summary: dict, args: argparse.Namespace) -> float:
    return (
        float(summary['mean_delta_pp'])
        + args.lcb_weight * float(summary['lcb90_delta_pp'])
        - args.std_penalty * float(summary['std_delta_pp'])
        - args.downside_penalty * float(summary['mean_downside_pp'])
        - args.max_downside_penalty * float(summary['max_downside_pp'])
    )


def save_state(args: argparse.Namespace, candidates: list[dict], complete: bool, round_index: int, survivors: list[int]) -> None:
    results = []
    for candidate in candidates:
        rows = candidate.get('rows', [])
        summary = summarize_rows(rows)
        results.append({
            **{k: v for k, v in candidate.items() if k not in {'decode_cfg', 'events', 'ids', 'frames', 'prompt_ids', 'rows'}},
            'rows': rows,
            'summary': summary,
            'risk_utility': utility(summary, args),
            'status': candidate.get('status', 'active'),
        })
    results = sorted(results, key=lambda row: row['risk_utility'], reverse=True)
    save_json({
        'complete': complete,
        'checkpoint': args.checkpoint,
        'candidates_json': args.candidates_json,
        'round_index': round_index,
        'rounds': parse_ints(args.rounds),
        'survivors': survivors,
        'scoring': {
            'lcb_weight': args.lcb_weight,
            'std_penalty': args.std_penalty,
            'downside_penalty': args.downside_penalty,
            'max_downside_penalty': args.max_downside_penalty,
            'hard_drop_delta_pp': args.hard_drop_delta_pp,
        },
        'results': results,
    }, args.out_json)


def main() -> None:
    parser = argparse.ArgumentParser(description='Risk-aware successive-halving validation for intervention candidates.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--candidates-json', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--top-k', type=int, default=16)
    parser.add_argument('--rounds', default='2,4,8')
    parser.add_argument('--keep-sizes', default='8,4,2')
    parser.add_argument('--seed', type=int, default=2000000)
    parser.add_argument('--step-minutes', type=int, default=5)
    parser.add_argument('--top-p', type=float, default=DecodeConfig.top_p)
    parser.add_argument('--temperature', type=float, default=DecodeConfig.temperature)
    parser.add_argument('--prompt-tail-tokens', type=int, default=DecodeConfig.prompt_tail_tokens)
    parser.add_argument('--prefix-possession-target', action='store_true')
    parser.add_argument('--possession-lookback-minutes', type=int, default=15)
    parser.add_argument('--external-possession-weight', type=float, default=0.0)
    parser.add_argument('--possession-target-calibration-json', default=None)
    parser.add_argument('--density-from-prefix', action='store_true')
    parser.add_argument('--density-lookback-minutes', type=int, default=15)
    parser.add_argument('--density-multiplier', type=float, default=1.0)
    parser.add_argument('--event-density-tolerance', type=int, default=DecodeConfig.event_density_tolerance)
    parser.add_argument('--conditional-prior-json', default=None)
    parser.add_argument('--conditional-prior-bias-strength', type=float, default=0.0)
    parser.add_argument('--no-organic-actions', action='store_true')
    parser.add_argument('--possession-guidance-prior-events', type=float, default=0.0)
    parser.add_argument('--reject-possession-below', type=float, default=0.02)
    parser.add_argument('--reject-possession-above', type=float, default=0.98)
    parser.add_argument('--min-possession-frames', type=int, default=8)
    parser.add_argument('--min-events-per-minute', type=float, default=10.0)
    parser.add_argument('--max-events-per-minute', type=float, default=80.0)
    parser.add_argument('--hard-drop-delta-pp', type=float, default=-6.0)
    parser.add_argument('--lcb-weight', type=float, default=0.50)
    parser.add_argument('--std-penalty', type=float, default=0.25)
    parser.add_argument('--downside-penalty', type=float, default=0.70)
    parser.add_argument('--max-downside-penalty', type=float, default=0.20)
    args = parser.parse_args()

    rounds = parse_ints(args.rounds)
    keep_sizes = parse_ints(args.keep_sizes)
    if len(keep_sizes) != len(rounds):
        raise RuntimeError('keep-sizes must have the same length as rounds')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get('token_cfg', {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    conditional_prior = load_json(args.conditional_prior_json) if args.conditional_prior_json else None
    possession_target_calibration = load_json(args.possession_target_calibration_json) if args.possession_target_calibration_json else None

    source = load_json(args.candidates_json)
    raw_candidates = select_candidates(source, args.top_k)
    candidates = []
    for candidate_index, raw in enumerate(raw_candidates):
        item = dict(raw)
        item['candidate_index'] = candidate_index
        item['match'] = Path(item['match_json']).stem
        item['rows'] = []
        item['status'] = 'active'
        events = load_json(item['match_json'])
        ids, frames = encoder.encode_match(events, None)
        cut_minute = int(item['cut_minute'])
        item['prompt_ids'] = ids[: token_index_at_cut(frames, cut_minute)]
        target_home_possession = None
        if args.prefix_possession_target:
            target_home_possession = prefix_home_possession_target(frames, cut_minute, args.possession_lookback_minutes)
            if target_home_possession is not None and possession_target_calibration:
                full_prefix_frames = [frame for frame in frames if 0 <= frame.get('minute', 0) < cut_minute]
                full_prefix_possession = home_possession_ratio(full_prefix_frames)
                target_home_possession = (
                    float(possession_target_calibration.get('intercept', 0.0))
                    + float(possession_target_calibration.get('prefix_coef', 1.0)) * float(target_home_possession)
                    + float(possession_target_calibration.get('full_prefix_coef', 0.0)) * float(full_prefix_possession)
                )
                target_home_possession = max(0.05, min(0.95, target_home_possession))
        target_events_per_minute = None
        if args.density_from_prefix:
            lookback_start = max(0, cut_minute - args.density_lookback_minutes)
            density_prefix_frames = [frame for frame in frames if lookback_start <= frame.get('minute', 0) < cut_minute]
            target_events_per_minute = (len(density_prefix_frames) / max(1, cut_minute - lookback_start)) * max(0.1, float(args.density_multiplier))
        item['decode_cfg'] = make_decode_cfg(args, target_home_possession, target_events_per_minute, conditional_prior)
        candidates.append(item)

    active = list(range(len(candidates)))
    for round_index, target_seeds in enumerate(rounds):
        for candidate_index in list(active):
            candidate = candidates[candidate_index]
            if candidate.get('status') != 'active':
                continue
            cut_minute = int(candidate['cut_minute'])
            end_minute = int(candidate.get('end_minute', cut_minute + args.step_minutes))
            step_minutes = int(candidate.get('step_minutes', args.step_minutes))
            action = candidate.get('action')
            n_steps = max(1, (end_minute - cut_minute) // step_minutes)
            policy_actions = [action] + ['ACTION::NOOP'] * (n_steps - 1)
            noop_actions = ['ACTION::NOOP'] * n_steps
            while len(candidate['rows']) < target_seeds:
                seed_index = len(candidate['rows'])
                seed = args.seed + candidate_index * 10000 + seed_index * 1000
                noop, _, noop_metrics = replay_with_metrics(
                    model, vocab, candidate['prompt_ids'], noop_actions, cut_minute, step_minutes,
                    token_cfg, candidate['decode_cfg'], device, seed,
                )
                policy, _, policy_metrics = replay_with_metrics(
                    model, vocab, candidate['prompt_ids'], policy_actions, cut_minute, step_minutes,
                    token_cfg, candidate['decode_cfg'], device, seed,
                )
                noop_reasons = sanity_reasons(noop_metrics, args)
                policy_reasons = sanity_reasons(policy_metrics, args)
                row = {
                    'seed': seed,
                    'noop_possession': noop,
                    'policy_possession': policy,
                    'delta_pp': (policy - noop) * 100.0,
                    'noop_metrics': noop_metrics,
                    'policy_metrics': policy_metrics,
                    'accepted': not noop_reasons and not policy_reasons,
                    'noop_reject_reasons': noop_reasons,
                    'policy_reject_reasons': policy_reasons,
                }
                candidate['rows'].append(row)
                print(
                    f"[progress] round={round_index + 1}/{len(rounds)} candidate={candidate_index + 1}/{len(candidates)} "
                    f"match={candidate['match']} minute={cut_minute} action={action} "
                    f"seed={seed_index + 1}/{target_seeds} delta_pp={row['delta_pp']:.3f} accepted={row['accepted']}",
                    flush=True,
                )
                if row['accepted'] and row['delta_pp'] <= args.hard_drop_delta_pp:
                    candidate['status'] = 'hard_dropped'
                    print(
                        f"[drop] candidate={candidate_index + 1}/{len(candidates)} reason=hard_downside "
                        f"delta_pp={row['delta_pp']:.3f}",
                        flush=True,
                    )
                    break
        ranked_active = []
        for candidate_index in active:
            candidate = candidates[candidate_index]
            summary = summarize_rows(candidate['rows'])
            candidate['last_summary'] = summary
            candidate['last_utility'] = utility(summary, args)
            if candidate.get('status') == 'active' and summary['n'] > 0:
                ranked_active.append(candidate_index)
        ranked_active = sorted(ranked_active, key=lambda idx: candidates[idx]['last_utility'], reverse=True)
        keep = min(keep_sizes[round_index], len(ranked_active))
        survivors = ranked_active[:keep]
        for candidate_index in ranked_active[keep:]:
            candidates[candidate_index]['status'] = 'halved'
        active = survivors
        print(
            f"[round_done] round={round_index + 1}/{len(rounds)} target_seeds={target_seeds} survivors={len(active)} "
            f"top={[candidates[idx]['match'] + '@' + str(candidates[idx]['cut_minute']) + ':' + str(round(candidates[idx]['last_utility'], 3)) for idx in active[:5]]}",
            flush=True,
        )
        save_state(args, candidates, complete=False, round_index=round_index + 1, survivors=active)
        if not active:
            break

    save_state(args, candidates, complete=True, round_index=len(rounds), survivors=active)


if __name__ == '__main__':
    main()
