import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from .config import DataConfig, TokenConfig, TrainConfig
from .dataset import RandomCutForwardDataset, collate_forward
from .model import compute_losses, load_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import mean, save_json
from .vocabulary import Vocab


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate supervised heads on the exact checkpoint holdout split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", default="impactful_possession_pipeline/runs/holdout_heads.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap on checkpoint test files; 0 means all.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, vocab_obj, model_cfg, extra = load_checkpoint(args.checkpoint, device)
    if "test_files" not in extra or "data_cfg" not in extra:
        raise RuntimeError("Checkpoint does not include test_files/data_cfg metadata.")

    vocab = Vocab.from_dict(vocab_obj)
    encoder = StructuredStatsBombEncoder(vocab, TokenConfig())
    data_cfg = DataConfig(**extra["data_cfg"])
    test_files = list(extra["test_files"])
    if args.max_files > 0:
        test_files = test_files[: args.max_files]

    dataset = RandomCutForwardDataset(test_files, data_cfg.lineups_dir, encoder, data_cfg, model_cfg.max_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_forward(batch, encoder.pad_id),
    )

    totals = {"loss": [], "token_loss": [], "possession_mae_pp": [], "count_loss": [], "uncertainty_loss": []}
    horizon_abs = defaultdict(list)
    count_abs = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to(device)
            outputs = model(batch["x"], batch["mask"], batch["context"], batch["context_mask"])
            losses = compute_losses(
                outputs,
                batch,
                pad_id=encoder.pad_id,
                possession_weight=TrainConfig.possession_weight,
                count_weight=TrainConfig.count_weight,
                uncertainty_weight=TrainConfig.uncertainty_weight,
            )
            for key in totals:
                totals[key].append(float(losses[key].detach().cpu()))
            pred_pos = torch.sigmoid(outputs["possession_logits"])
            targets = torch.stack([batch["target_pos_5"], batch["target_pos_10"], batch["target_pos_rest"]], dim=1)
            for idx, name in enumerate(("5m", "10m", "rest")):
                horizon_abs[name].extend(((pred_pos[:, idx] - targets[:, idx]).abs() * 100.0).detach().cpu().tolist())
            pred_counts = torch.expm1(outputs["count_logs"]).clamp_min(0.0)
            counts = torch.stack([batch["target_count_5"], batch["target_count_10"]], dim=1)
            for idx, name in enumerate(("5m", "10m")):
                count_abs[name].extend((pred_counts[:, idx] - counts[:, idx]).abs().detach().cpu().tolist())

    summary = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "holdout_files": len(test_files),
        "holdout_examples": len(dataset),
        **{key: mean(values) for key, values in totals.items()},
        "possession_mae_pp_by_horizon": {key: mean(values) for key, values in horizon_abs.items()},
        "count_mae_by_horizon": {key: mean(values) for key, values in count_abs.items()},
    }
    save_json({"summary": summary}, args.out_json)
    print(f"saved holdout head evaluation to {args.out_json}")
    print(summary)


if __name__ == "__main__":
    main()
