import argparse
from pathlib import Path

import torch

from .config import DataConfig, DecodeConfig, PlannerConfig, TokenConfig
from .dataset import frames_after_cut, token_index_at_cut
from .decode import decode_frames, generate_continuation
from .model import load_checkpoint
from .planner import plan_second_half
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import event_type_counts, home_possession_ratio, list_event_files, load_json, mean, save_json, split_files
from .vocabulary import Vocab


def resolve_test_files(checkpoint_extra: dict, args) -> list[str]:
    """Use the exact held-out match list the checkpoint was trained against.

    Recomputing the split at eval time with a different --max-matches silently
    shifts the train/val/test boundaries (they scale with len(paths)) even
    though the underlying shuffle permutation is unchanged, so a "test" slice
    recomputed with a smaller max_matches can land entirely inside the real
    training set. Always prefer the persisted list; only fall back to
    recomputation (with a loud warning) for older checkpoints that predate
    saving it.
    """
    persisted = checkpoint_extra.get("test_files")
    if persisted:
        if args.max_matches < len(persisted):
            print(
                f"NOTE: checkpoint has {len(persisted)} held-out test matches; "
                f"using the first {args.max_matches} of that exact persisted list.",
                flush=True,
            )
        return persisted[: args.max_matches]
    print(
        "WARNING: checkpoint has no persisted test_files (older checkpoint). "
        "Recomputing the split from scratch with --max-matches "
        f"{args.max_matches} — this is ONLY correct if it matches the "
        "max_matches used at training time, otherwise the recomputed split "
        "boundaries will not line up with the true held-out set.",
        flush=True,
    )
    data_cfg = DataConfig()
    files = list_event_files(args.events_dir, args.max_matches, args.seed)
    _, _, test_files = split_files(files, data_cfg.train_fraction, data_cfg.val_fraction)
    return test_files[: args.max_matches]


def evaluate_match(model, vocab, encoder, event_file, lineups_dir, device, seed, run_planner, decode_cfg):
    events = load_json(event_file)
    lineup_path = encoder.lineup_path_for_event_file(event_file, lineups_dir)
    ids, frames = encoder.encode_match(events, lineup_path)
    cut = 45
    cut_idx = token_index_at_cut(frames, cut)
    prompt_ids = ids[:cut_idx]
    real_frames = frames_after_cut(frames, cut)
    real_pos = home_possession_ratio(real_frames)

    generated = generate_continuation(
        model,
        vocab,
        prompt_ids,
        start_minute=cut,
        end_minute=90,
        token_cfg=TokenConfig(),
        decode_cfg=decode_cfg,
        device=device,
        seed=seed,
    )
    sim_frames = decode_frames(vocab, generated, start_minute=cut)
    sim_pos = home_possession_ratio(sim_frames)

    real_counts = event_type_counts(real_frames)
    sim_counts = event_type_counts(sim_frames)
    real_passes, sim_passes = real_counts.get("Pass", 0), sim_counts.get("Pass", 0)
    real_shots, sim_shots = real_counts.get("Shot", 0), sim_counts.get("Shot", 0)
    real_dribbles, sim_dribbles = real_counts.get("Dribble", 0), sim_counts.get("Dribble", 0)
    real_total, sim_total = sum(real_counts.values()), sum(sim_counts.values())

    row = {
        "match": Path(event_file).name,
        "real_home_possession": real_pos,
        "no_intervention_home_possession": sim_pos,
        "abs_error_pp": abs(sim_pos - real_pos) * 100.0,
        "generated_frame_count": len(sim_frames),
        "real_passes": real_passes,
        "sim_passes": sim_passes,
        "passes_abs_error": abs(sim_passes - real_passes),
        "real_shots": real_shots,
        "sim_shots": sim_shots,
        "shots_abs_error": abs(sim_shots - real_shots),
        "real_dribbles": real_dribbles,
        "sim_dribbles": sim_dribbles,
        "dribbles_abs_error": abs(sim_dribbles - real_dribbles),
        "real_total_events": real_total,
        "sim_total_events": sim_total,
        "total_events_abs_error": abs(sim_total - real_total),
    }
    if run_planner:
        plan = plan_second_half(
            model,
            vocab,
            prompt_ids,
            token_cfg=TokenConfig(),
            decode_cfg=decode_cfg,
            planner_cfg=PlannerConfig(),
            device=device,
            seed=seed,
        )
        row["planned_home_possession"] = plan["simulated_home_possession"]
        row["planned_delta_vs_noop"] = plan["simulated_home_possession"] - sim_pos
        row["interventions"] = plan["interventions"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simulator accuracy and optional intervention planning.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--max-matches", type=int, default=100)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/evaluation.json")
    parser.add_argument("--run-planner", action="store_true")
    parser.add_argument("--decode-min-events-per-minute", type=int, default=DecodeConfig.min_events_per_minute)
    parser.add_argument("--decode-max-events-per-minute", type=int, default=DecodeConfig.max_events_per_minute)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_obj, _, checkpoint_extra = load_checkpoint(args.checkpoint, device)
    vocab = Vocab.from_dict(vocab_obj)
    encoder = StructuredStatsBombEncoder(vocab, TokenConfig())
    decode_cfg = DecodeConfig(min_events_per_minute=args.decode_min_events_per_minute, max_events_per_minute=args.decode_max_events_per_minute)
    test_files = resolve_test_files(checkpoint_extra, args)
    rows = [
        evaluate_match(model, vocab, encoder, path, args.lineups_dir, device, args.seed + i, args.run_planner, decode_cfg)
        for i, path in enumerate(test_files)
    ]
    summary = {
        "num_matches": len(rows),
        "no_intervention_mae_pp": mean([r["abs_error_pp"] for r in rows]),
        "mean_generated_frame_count": mean([r["generated_frame_count"] for r in rows]),
        "passes_mae": mean([r["passes_abs_error"] for r in rows]),
        "shots_mae": mean([r["shots_abs_error"] for r in rows]),
        "dribbles_mae": mean([r["dribbles_abs_error"] for r in rows]),
        "total_events_mae": mean([r["total_events_abs_error"] for r in rows]),
    }
    if args.run_planner:
        summary["mean_planned_delta_vs_noop_pp"] = mean([r["planned_delta_vs_noop"] * 100.0 for r in rows])
    save_json({"summary": summary, "matches": rows}, args.out_json)
    print(f"saved evaluation to {args.out_json}")
    print(summary)


if __name__ == "__main__":
    main()
