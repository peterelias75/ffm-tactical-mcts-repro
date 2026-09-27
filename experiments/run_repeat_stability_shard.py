"""Resumable E04 repeated-simulation stability evaluation.

Each individual simulation is persisted immediately. This preserves work across
host interruptions and computes the paper's per-match population variance and
absolute error of the ten-run mean versus real second-half possession.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

from impactful_possession_pipeline.config import DecodeConfig, TokenConfig
from impactful_possession_pipeline.dataset import frames_after_cut, token_index_at_cut
from impactful_possession_pipeline.decode import decode_frames, generate_continuation
from impactful_possession_pipeline.evaluate_pipeline import resolve_test_files
from impactful_possession_pipeline.model import load_checkpoint
from impactful_possession_pipeline.statsbomb_features import StructuredStatsBombEncoder
from impactful_possession_pipeline.utils import home_possession_ratio, load_json, mean, save_json
from impactful_possession_pipeline.vocabulary import Vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events-dir", default="/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/events")
    parser.add_argument("--lineups-dir", default="/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/lineups")
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--runs-per-match", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42000)
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
    proxy = argparse.Namespace(max_matches=10**9, events_dir=args.events_dir, seed=42)
    selected = resolve_test_files(extra, proxy)[args.offset : args.offset + args.count]
    out_path = Path(args.out_json)
    records = load_json(out_path).get("simulations", []) if args.resume and out_path.exists() else []
    done = {(r["match"], int(r["run_index"])) for r in records}

    def summarize(complete: bool) -> dict:
        grouped = defaultdict(list)
        real = {}
        for record in records:
            grouped[record["match"]].append(record["simulated_home_possession"] * 100.0)
            real[record["match"]] = record["real_home_possession"] * 100.0
        match_rows = []
        for match, values in sorted(grouped.items()):
            avg = mean(values)
            squared_deviations = [(value - avg) ** 2 for value in values]
            population_variance = mean(squared_deviations)
            # The historical paper-generating script uses statistics.variance,
            # i.e. the N-1 sample variance. Preserve population variance too so
            # the denominator choice is explicit and independently auditable.
            sample_variance = (
                sum(squared_deviations) / (len(values) - 1)
                if len(values) > 1 else 0.0
            )
            match_rows.append({
                "match": match, "num_runs": len(values), "mean_simulated_possession_pp": avg,
                "sample_variance_pp2": sample_variance,
                "population_variance_pp2": population_variance,
                "absolute_error_pp": abs(avg - real[match]),
                "real_home_possession_pp": real[match],
            })
        complete_rows = [row for row in match_rows if row["num_runs"] == args.runs_per_match]
        return {
            "summary": {
                "offset": args.offset, "num_requested_matches": len(selected),
                "runs_per_match": args.runs_per_match, "num_saved_simulations": len(records),
                "num_complete_matches": len(complete_rows), "complete": complete,
                "mean_within_match_sample_variance_pp2": mean([r["sample_variance_pp2"] for r in complete_rows]),
                "mean_within_match_population_variance_pp2": mean([r["population_variance_pp2"] for r in complete_rows]),
                "mean_abs_error_of_simulation_mean_pp": mean([r["absolute_error_pp"] for r in complete_rows]),
                "paper_variance_target_pp2": 13.87, "paper_error_target_pp": 12.02,
            },
            "matches": match_rows,
            "simulations": records,
        }

    for match_offset, path in enumerate(selected):
        match = Path(path).name
        events = load_json(path)
        lineup_path = encoder.lineup_path_for_event_file(path, args.lineups_dir)
        ids, frames = encoder.encode_match(events, lineup_path)
        prompt_ids = ids[:token_index_at_cut(frames, 45)]
        real_pos = home_possession_ratio(frames_after_cut(frames, 45))
        for run_index in range(args.runs_per_match):
            if (match, run_index) in done:
                continue
            started = time.time()
            seed = args.seed + (args.offset + match_offset) * 1000 + run_index
            generated = generate_continuation(model, vocab, prompt_ids, 45, 90, token_cfg,
                                              DecodeConfig(), device, seed)
            generated_frames = decode_frames(vocab, generated, start_minute=45)
            sim_pos = home_possession_ratio(generated_frames)
            records.append({
                "match": match, "run_index": run_index, "seed": seed,
                "real_home_possession": real_pos, "simulated_home_possession": sim_pos,
                "generated_frame_count": len(generated_frames),
                "elapsed_seconds": time.time() - started,
            })
            save_json(summarize(False), out_path)
            print(f"{match} run={run_index + 1}/{args.runs_per_match} sim={sim_pos * 100:.2f}%", flush=True)
    save_json(summarize(True), out_path)


if __name__ == "__main__":
    main()
