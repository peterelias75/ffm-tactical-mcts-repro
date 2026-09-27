"""Random and Classic-Markov baseline policies for the Table 2 comparison.

Mirrors evaluate_mcts_pipeline.py's structure and output schema exactly (same
summary fields: mean_delta_pp, pct_matches_improved, mean_gain_when_improved_pp,
bootstrap_lcb90_pp) so all four Table 2 policies are directly comparable. The
only thing that differs between this script and evaluate_mcts_pipeline.py is
the decision rule at each 5-minute decision point:

  - random: uniformly sample one legal action, no rollout/search.
  - classic_markov: deterministic lookup in a fixed table of historical
    action -> possession-delta estimates (classic_markov.py), no per-match
    adaptation and no rollout/search.

Both still advance the match with the same forward-model generation used by
every other policy, so the only thing being compared is the decision rule.
"""

import argparse
import random as py_random
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, PlannerConfig, TokenConfig
from .classic_markov import classic_markov_decide
from .dataset import frames_after_cut, token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .planner import action_kind, legal_actions
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import bootstrap_mean_lcb, home_possession_ratio, load_json, mean, save_json, std
from .vocabulary import Vocab


def run_policy_second_half(
    model, vocab, prompt_ids, token_cfg, decode_cfg, planner_cfg, device, seed, cut_minute, end_minute,
    policy: str, table: dict | None,
) -> dict:
    ctx = list(prompt_ids)
    interventions = []
    used_subs = 0
    used_tactics = 0
    cooldown = 0
    rng = py_random.Random(seed)
    for minute in range(cut_minute, end_minute, planner_cfg.step_minutes):
        legal = legal_actions(vocab, used_subs, used_tactics, cooldown, planner_cfg, token_cfg)
        if policy == "random":
            action = rng.choice(legal)
        elif policy == "classic_markov":
            action = classic_markov_decide(table, legal)
        else:
            raise ValueError(f"unknown policy {policy}")

        if action != "ACTION::NOOP":
            ctx.append(vocab.id(action))
            if action_kind(action) == "sub":
                used_subs += 1
            elif action_kind(action) == "tactic":
                used_tactics += 1
                cooldown = planner_cfg.tactical_cooldown_steps
        else:
            cooldown = max(0, cooldown - 1)

        generated = generate_continuation(
            model, vocab, ctx, minute, min(end_minute, minute + planner_cfg.step_minutes),
            token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=seed + minute + 500_000,
        )
        eos_id = vocab.id("<eos>")
        ctx.extend([tok for tok in generated if tok != eos_id])
        interventions.append({"minute": minute, "chosen_action": action})

    frames = decode_frames(vocab, ctx, start_minute=0)
    rest_frames = [f for f in frames if cut_minute <= f.get("minute", 0) < end_minute]
    return {
        "interventions": interventions,
        "simulated_home_possession": home_possession_ratio(rest_frames),
        "used_subs": used_subs,
        "used_tactical_shifts": used_tactics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["random", "classic_markov"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--holdout-json", default="impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json")
    parser.add_argument("--markov-table-json", default="impactful_possession_pipeline/runs/classic_markov_table.json")
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--cut-minute", type=int, default=45)
    parser.add_argument("--step-minutes", type=int, default=PlannerConfig.step_minutes)
    parser.add_argument("--allow-defensive-actions", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.84, device=0)
    model, vocab_obj, _, extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    token_cfg = TokenConfig(**extra.get("token_cfg", {}))
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)
    decode_cfg = DecodeConfig(temperature=args.temperature, top_p=args.top_p)
    planner_cfg = PlannerConfig(step_minutes=args.step_minutes, allow_defensive_actions=args.allow_defensive_actions)

    table = None
    if args.policy == "classic_markov":
        table = load_json(args.markov_table_json)

    holdout = load_json(args.holdout_json)
    match_ids = holdout["match_ids"]
    if args.max_matches:
        match_ids = match_ids[: args.max_matches]

    out_path = Path(args.out_json)
    rows = []
    done_matches = set()
    if args.resume and out_path.exists():
        partial = load_json(out_path)
        rows = list(partial.get("matches", []))
        done_matches = {str(r.get("match")) for r in rows}
        print(f"resuming from {out_path}: {len(done_matches)} completed matches", flush=True)

    def write_state(complete: bool) -> None:
        deltas = [r["delta_pp"] for r in rows]
        positive = [d for d in deltas if d > 0]
        summary = {
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "holdout_json": args.holdout_json,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_matches": len(rows),
            "num_requested_matches": len(match_ids),
            "step_minutes": args.step_minutes,
            "mean_delta_pp": mean(deltas),
            "std_delta_pp": std(deltas),
            "pct_matches_improved": 100.0 * mean([1.0 if d > 0 else 0.0 for d in deltas]),
            "mean_gain_when_improved_pp": mean(positive) if positive else 0.0,
            "bootstrap_lcb90_pp": bootstrap_mean_lcb(deltas, alpha=0.10, seed=args.seed) if deltas else 0.0,
            "complete": complete,
        }
        save_json({"summary": summary, "matches": rows}, args.out_json)

    for i, match_id in enumerate(match_ids):
        if str(match_id) in done_matches:
            print(f"{i + 1}/{len(match_ids)} {match_id} already done; skipping", flush=True)
            continue

        event_file = str(Path(args.events_dir) / match_id)
        events = load_json(event_file)
        lineup_path = encoder.lineup_path_for_event_file(event_file, args.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        cut_idx = token_index_at_cut(frames, args.cut_minute)
        if cut_idx <= 2 or cut_idx >= len(ids) - 2:
            print(f"{i + 1}/{len(match_ids)} {match_id} skipped (cut index out of range)", flush=True)
            continue
        prompt_ids = ids[:cut_idx]
        seed = args.seed + i

        noop_generated = generate_continuation(
            model, vocab, prompt_ids, start_minute=args.cut_minute, end_minute=90,
            token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=seed,
        )
        noop_frames = decode_frames(vocab, noop_generated, start_minute=args.cut_minute)
        noop_pos = home_possession_ratio(noop_frames)

        result = run_policy_second_half(
            model, vocab, prompt_ids, token_cfg, decode_cfg, planner_cfg, device, seed,
            cut_minute=args.cut_minute, end_minute=90, policy=args.policy, table=table,
        )
        raw_pos = result["simulated_home_possession"]
        has_intervention = (result["used_subs"] + result["used_tactical_shifts"]) > 0
        policy_pos = raw_pos if has_intervention else noop_pos
        delta = policy_pos - noop_pos
        real_pos = home_possession_ratio(frames_after_cut(frames, args.cut_minute))

        row = {
            "match": match_id,
            "real_home_possession": real_pos,
            "no_intervention_home_possession": noop_pos,
            "policy_home_possession": policy_pos,
            "delta_pp": delta * 100.0,
            "used_subs": result["used_subs"],
            "used_tactical_shifts": result["used_tactical_shifts"],
            "interventions": result["interventions"],
        }
        rows.append(row)
        done_matches.add(str(match_id))
        write_state(complete=False)
        print(
            f"{i + 1}/{len(match_ids)} {match_id} noop={noop_pos:.3f} {args.policy}={policy_pos:.3f} "
            f"delta={delta * 100:+.2f}pp subs={row['used_subs']} tactics={row['used_tactical_shifts']}",
            flush=True,
        )

    write_state(complete=True)
    print(f"saved to {args.out_json}", flush=True)
    print(load_json(args.out_json)["summary"], flush=True)


if __name__ == "__main__":
    main()
