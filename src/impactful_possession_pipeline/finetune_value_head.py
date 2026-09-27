"""Fine-tune only the value/count/uncertainty heads with a recency-windowed pool.

Diagnosed root cause (see runs/full_holdout_checks/phase2_v3_final_report.md):
ForwardSimulatorModel.pooled_context used to mean-pool the ENTIRE context, so a single
appended action token's unique hidden state -- the only thing causal attention lets differ
between a no-op and an action continuation -- got diluted to roughly 1/context_length,
making the value head structurally unable to tell which action was taken. model.py now
supports ModelConfig.value_pool_window to pool only the last N valid tokens instead, which
keeps the action token's relative weight high regardless of match length.

This script freezes the already-trained backbone (token/position embeddings, transformer
blocks, final layer norm, and the language-model head -- none of that needs to change, it
already produces reasonable next-token predictions and ~7.4pp free-running rollout MAE) and
only trains freshly-initialized value/count/uncertainty heads on top of the new windowed
pooling. The old heads are NOT reused as a warm start: they were fit to the statistics of a
full-sequence mean-pooled input, which is a different distribution than a short recency
window, so warm-starting from them could anchor optimization in a bad basin.

Because the backbone forward pass runs under torch.no_grad(), this is far cheaper than a
full retrain -- only two small MLPs are actually being optimized.

Usage:
    python3 -m impactful_possession_pipeline.finetune_value_head \
        --base-checkpoint impactful_possession_pipeline/runs/phase1_v3_richactions_24epochs_20260722/epoch_checkpoints/epoch016.pt \
        --out-dir impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64 \
        --value-pool-window 64 --epochs 8
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig, TokenConfig
from .dataset import RandomCutForwardDataset, collate_forward
from .model import ForwardSimulatorModel, save_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import list_event_files, mean, seed_everything, split_files
from .vocabulary import Vocab

BACKBONE_PREFIXES = ("token_embedding.", "position_embedding.", "blocks.", "ln_f.", "lm_head.")


def is_backbone_param(name: str) -> bool:
    return name.startswith(BACKBONE_PREFIXES)


def value_losses(model: ForwardSimulatorModel, batch: dict, possession_weight: float, count_weight: float, uncertainty_weight: float) -> dict:
    with torch.no_grad(), torch.autocast(device_type=batch["context"].device.type, enabled=batch["context"].is_cuda):
        pooled = model.pooled_context(batch["context"], batch["context_mask"]).float()
    pos_logits = model.value_head(pooled)
    count_logits = model.count_head(pooled)
    uncertainty_logits = model.uncertainty_head(pooled)

    targets = torch.stack([batch["target_pos_5"], batch["target_pos_10"], batch["target_pos_rest"]], dim=1)
    pred_pos = torch.sigmoid(pos_logits)
    abs_err = (pred_pos - targets).abs()
    possession_loss = abs_err.mean()

    counts = torch.stack([batch["target_count_5"], batch["target_count_10"]], dim=1)
    count_loss = F.smooth_l1_loss(count_logits, torch.log1p(counts))

    uncertainty_target = torch.log(abs_err.detach().clamp_min(1e-4))
    uncertainty_loss = F.smooth_l1_loss(uncertainty_logits, uncertainty_target)

    total = possession_weight * possession_loss + count_weight * count_loss + uncertainty_weight * uncertainty_loss
    return {
        "loss": total,
        "possession_mae_pp": (abs_err.mean() * 100.0).detach(),
        "count_loss": count_loss.detach(),
        "uncertainty_loss": uncertainty_loss.detach(),
    }


def run_epoch(model, loader, optimizer, cfg_weights, device, train: bool) -> dict:
    model.train(train)
    totals = {"loss": [], "possession_mae_pp": [], "count_loss": [], "uncertainty_loss": []}
    phase = "train" if train else "val"
    for step, batch in enumerate(loader, start=1):
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        with torch.set_grad_enabled(train):
            losses = value_losses(model, batch, **cfg_weights)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key].append(float(losses[key].detach().cpu()))
        if step % 500 == 0 or step == len(loader):
            print(f"{phase} step={step}/{len(loader)} loss={float(losses['loss'].detach().cpu()):.4f} pos_mae={float(losses['possession_mae_pp'].detach().cpu()):.2f}pp", flush=True)
    return {key: mean(vals) for key, vals in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, help="Trained forward-simulator checkpoint to freeze the backbone from.")
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--value-pool-window", type=int, default=64, help="Pool only the last N valid context tokens for the value/count/uncertainty heads, instead of the whole match.")
    parser.add_argument("--max-matches", type=int, default=DataConfig.max_matches)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8, help="Kept modest despite the frozen no_grad backbone: attention memory is O(seq_len^2) and some matches produce contexts near max_len, so a worst-case batch can still spike GPU memory.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--possession-weight", type=float, default=1.0)
    parser.add_argument("--count-weight", type=float, default=0.15)
    parser.add_argument("--uncertainty-weight", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    old_cfg = ModelConfig(**ckpt["model_cfg"])
    vocab = Vocab.from_dict(ckpt["vocab"])
    token_cfg = TokenConfig(**ckpt["extra"].get("token_cfg", {}))

    new_cfg_dict = dict(old_cfg.__dict__)
    new_cfg_dict["value_pool_window"] = args.value_pool_window
    new_cfg = ModelConfig(**new_cfg_dict)

    model = ForwardSimulatorModel(len(vocab), new_cfg).to(device)
    old_state = ckpt["model"]
    new_state = model.state_dict()
    copied = 0
    for key, value in old_state.items():
        if is_backbone_param(key) and key in new_state and tuple(new_state[key].shape) == tuple(value.shape):
            new_state[key] = value
            copied += 1
    model.load_state_dict(new_state, strict=True)
    print(f"copied {copied} backbone tensors from {args.base_checkpoint}; value/count/uncertainty heads freshly initialized (window={args.value_pool_window})", flush=True)

    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if is_backbone_param(name):
            param.requires_grad = False
            frozen += param.numel()
        else:
            trainable += param.numel()
    print(f"frozen params={frozen:,} trainable params={trainable:,}", flush=True)

    data_cfg = DataConfig(events_dir=args.events_dir, lineups_dir=args.lineups_dir, max_matches=args.max_matches)
    seed_everything(data_cfg.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_event_files(data_cfg.events_dir, data_cfg.max_matches, data_cfg.seed)
    train_files, val_files, test_files = split_files(files, data_cfg.train_fraction, data_cfg.val_fraction)
    print(f"files train={len(train_files)} val={len(val_files)} test={len(test_files)}", flush=True)
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)

    train_ds = RandomCutForwardDataset(train_files, data_cfg.lineups_dir, encoder, data_cfg, new_cfg.max_len, progress_every=200, progress_label="train")
    val_ds = RandomCutForwardDataset(val_files, data_cfg.lineups_dir, encoder, data_cfg, new_cfg.max_len, progress_every=200, progress_label="val")
    print(f"train examples={len(train_ds)} val examples={len(val_ds)}", flush=True)

    collate = lambda batch: collate_forward(batch, encoder.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("NOTE: multiple GPUs visible but this script intentionally runs single-GPU (frozen backbone makes it fast enough).", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    cfg_weights = {
        "possession_weight": args.possession_weight,
        "count_weight": args.count_weight,
        "uncertainty_weight": args.uncertainty_weight,
    }

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, cfg_weights, device, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, cfg_weights, device, train=False)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_pos_mae={val_metrics['possession_mae_pp']:.2f}pp",
            flush=True,
        )
        if val_metrics["possession_mae_pp"] < best_mae:
            best_mae = val_metrics["possession_mae_pp"]
            save_checkpoint(
                str(out_dir / "best_forward_simulator.pt"),
                model,
                vocab.to_dict(),
                new_cfg,
                {
                    "token_cfg": token_cfg.__dict__,
                    "data_cfg": data_cfg.__dict__,
                    "best_val_possession_mae_pp": best_mae,
                    "test_files": ckpt["extra"].get("test_files", test_files),
                    "base_checkpoint": args.base_checkpoint,
                    "value_pool_window": args.value_pool_window,
                },
            )
            print(f"saved best value-head checkpoint with val_pos_mae={best_mae:.2f}pp", flush=True)


if __name__ == "__main__":
    main()
