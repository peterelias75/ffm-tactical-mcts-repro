import argparse
import math
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import DataConfig, DecodeConfig, ModelConfig, TokenConfig, TrainConfig
from .dataset import RandomCutForwardDataset, collate_forward
from .evaluate_pipeline import evaluate_match
from .model import ForwardSimulatorModel, compute_losses, save_checkpoint
from .statsbomb_features import StructuredStatsBombEncoder
from .utils import list_event_files, load_json, mean, seed_everything, split_files
from .vocabulary import Vocab, build_vocab


def cosine_warmup_lr_lambda(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def rollout_holdout_mae(model, vocab, encoder, match_files, lineups_dir, device, decode_cfg, cut_minute: int = 45, seed: int = 20260719) -> float:
    """Free-running generation MAE on a fixed small subset of the pinned holdout.

    This is the metric the planner actually depends on (unlike teacher-forced
    val_pos_mae), so we track it periodically even though it's slower to
    compute, to avoid repeating the mistake of only ever looking at the easy
    metric while the number that matters stays untracked.
    """
    bare_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    bare_model.eval()
    errors = []
    for i, path in enumerate(match_files):
        row = evaluate_match(bare_model, vocab, encoder, path, lineups_dir, device, seed + i, run_planner=False, decode_cfg=decode_cfg)
        errors.append(row["abs_error_pp"])
    bare_model.train()
    return mean(errors)


def load_compatible_checkpoint(path: str, model: ForwardSimulatorModel, new_vocab: Vocab, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    old_vocab = Vocab.from_dict(ckpt["vocab"])
    old_cfg = ModelConfig(**ckpt["model_cfg"])
    old_model = ForwardSimulatorModel(len(old_vocab), old_cfg).to(device)
    old_model.load_state_dict(ckpt["model"], strict=True)
    old_state = old_model.state_dict()
    new_state = model.state_dict()
    copied = []
    for key, value in old_state.items():
        if key not in new_state:
            continue
        if key in {"token_embedding.weight", "lm_head.weight"}:
            target = new_state[key].clone()
            for token, old_idx in old_vocab.stoi.items():
                new_idx = new_vocab.stoi.get(token)
                if new_idx is not None and old_idx < value.size(0) and new_idx < target.size(0):
                    target[new_idx].copy_(value[old_idx])
            new_state[key] = target
            copied.append(key)
        elif tuple(value.shape) == tuple(new_state[key].shape):
            new_state[key] = value
            copied.append(key)
    model.load_state_dict(new_state, strict=True)
    print(f"compatible resume copied {len(copied)}/{len(new_state)} tensors from {path}", flush=True)
    print(f"old_vocab={len(old_vocab)} new_vocab={len(new_vocab)} added_tokens={len(new_vocab) - len(old_vocab)}", flush=True)
    return ckpt.get("extra", {})


def run_epoch(model, loader, optimizer, scaler, cfg: TrainConfig, pad_id: int, device: torch.device, train: bool) -> dict:
    model.train(train)
    totals = {"loss": [], "token_loss": [], "possession_mae_pp": [], "count_loss": [], "uncertainty_loss": []}
    phase = "train" if train else "val"
    for step, batch in enumerate(loader, start=1):
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        with torch.set_grad_enabled(train):
            with autocast(enabled=cfg.amp and device.type == "cuda"):
                outputs = model(batch["x"], batch["mask"], batch["context"], batch["context_mask"])
                losses = compute_losses(
                    outputs,
                    batch,
                    pad_id=pad_id,
                    possession_weight=cfg.possession_weight,
                    count_weight=cfg.count_weight,
                    uncertainty_weight=cfg.uncertainty_weight,
                )
            if train:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key].append(float(losses[key].detach().cpu()))
        if step % 500 == 0 or step == len(loader):
            print(f"{phase} step={step}/{len(loader)} loss={float(losses['loss'].detach().cpu()):.4f}", flush=True)
    return {key: mean(vals) for key, vals in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-free random-cut football forward simulator.")
    parser.add_argument("--events-dir", default=DataConfig.events_dir)
    parser.add_argument("--lineups-dir", default=DataConfig.lineups_dir)
    parser.add_argument("--out-dir", default="impactful_possession_pipeline/runs/forward_simulator")
    parser.add_argument("--max-matches", type=int, default=DataConfig.max_matches)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--n-layers", type=int, default=ModelConfig.n_layers)
    parser.add_argument("--n-heads", type=int, default=ModelConfig.n_heads)
    parser.add_argument("--max-len", type=int, default=ModelConfig.max_len)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--possession-weight", type=float, default=TrainConfig.possession_weight)
    parser.add_argument("--count-weight", type=float, default=TrainConfig.count_weight)
    parser.add_argument("--uncertainty-weight", type=float, default=TrainConfig.uncertainty_weight)
    parser.add_argument("--event-token-weight", type=float, default=TrainConfig.event_token_weight)
    parser.add_argument("--team-token-weight", type=float, default=TrainConfig.team_token_weight)
    parser.add_argument("--role-token-weight", type=float, default=TrainConfig.role_token_weight)
    parser.add_argument("--location-token-weight", type=float, default=TrainConfig.location_token_weight)
    parser.add_argument("--possession-flip-token-weight", type=float, default=TrainConfig.possession_flip_token_weight)
    parser.add_argument("--goal-token-weight", type=float, default=TrainConfig.goal_token_weight)
    parser.add_argument("--reset-best-metric", action="store_true", help="When resuming, ignore checkpoint best metric for this output directory.")
    parser.add_argument("--rich-actions", action="store_true", help="Encode role/formation-specific intervention tokens.")
    parser.add_argument("--emit-formation-state", action="store_true", help="Encode periodic STATE::{TEAM}_FORM_* tokens (kickoff + every --formation-state-interval-minutes + after real tactical shifts), so the value head and generation have explicit current-formation context instead of relying on it leaking through raw event patterns.")
    parser.add_argument("--formation-state-interval-minutes", type=int, default=TokenConfig.formation_state_interval_minutes)
    parser.add_argument("--no-action-anchor-pairs", action="store_true", help="Disable extra pre/post-action training examples anchored to real subs/tactical shifts.")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Linear LR warmup epochs before cosine decay.")
    parser.add_argument("--rollout-eval-every-epochs", type=int, default=0, help="If >0, run free-running generation MAE on a pinned-holdout subset every N epochs (slow; the metric the planner actually needs).")
    parser.add_argument("--rollout-eval-matches", type=int, default=5, help="How many pinned-holdout matches to use for the periodic rollout MAE check.")
    parser.add_argument("--pinned-holdout-json", default="impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json")
    parser.add_argument("--start-epoch", type=int, default=1, help="Resume the LR schedule and epoch loop from this epoch (use with --resume-checkpoint after an interruption, e.g. a host reboot, so the cosine schedule and remaining epoch count pick up where they left off instead of restarting).")
    args = parser.parse_args()

    data_cfg = DataConfig(
        events_dir=args.events_dir,
        lineups_dir=args.lineups_dir,
        max_matches=args.max_matches,
        action_anchor_pairs=not args.no_action_anchor_pairs,
    )
    token_cfg = TokenConfig(
        rich_actions=args.rich_actions,
        emit_formation_state=args.emit_formation_state,
        formation_state_interval_minutes=args.formation_state_interval_minutes,
    )
    model_cfg = ModelConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, max_len=args.max_len)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        possession_weight=args.possession_weight,
        count_weight=args.count_weight,
        uncertainty_weight=args.uncertainty_weight,
        event_token_weight=args.event_token_weight,
        team_token_weight=args.team_token_weight,
        role_token_weight=args.role_token_weight,
        location_token_weight=args.location_token_weight,
        possession_flip_token_weight=args.possession_flip_token_weight,
        goal_token_weight=args.goal_token_weight,
    )
    seed_everything(data_cfg.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_event_files(data_cfg.events_dir, data_cfg.max_matches, data_cfg.seed)
    train_files, val_files, test_files = split_files(files, data_cfg.train_fraction, data_cfg.val_fraction)
    print(f"files train={len(train_files)} val={len(val_files)} test={len(test_files)}", flush=True)
    vocab = build_vocab(token_cfg)
    encoder = StructuredStatsBombEncoder(vocab, token_cfg)

    label_weights = {
        "event": train_cfg.event_token_weight,
        "team": train_cfg.team_token_weight,
        "role": train_cfg.role_token_weight,
        "location": train_cfg.location_token_weight,
        "possession_flip": train_cfg.possession_flip_token_weight,
        "goal": train_cfg.goal_token_weight,
    }
    print(f"label_weights={label_weights}", flush=True)
    print("building train dataset...", flush=True)
    train_ds = RandomCutForwardDataset(train_files, data_cfg.lineups_dir, encoder, data_cfg, model_cfg.max_len, label_weights=label_weights, progress_every=100, progress_label="train")
    anchor_counts: dict[str, int] = {}
    for ex in train_ds.examples:
        key = ex.get("anchor", "grid")
        anchor_counts[key] = anchor_counts.get(key, 0) + 1
    print(f"train examples={len(train_ds)} anchor_counts={anchor_counts} action_anchor_pairs={data_cfg.action_anchor_pairs}", flush=True)
    print("building val dataset...", flush=True)
    val_ds = RandomCutForwardDataset(val_files, data_cfg.lineups_dir, encoder, data_cfg, model_cfg.max_len, label_weights=label_weights, progress_every=100, progress_label="val")
    print(f"val examples={len(val_ds)}", flush=True)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("No training/validation examples were built. Check events-dir and cut minutes.")

    collate = lambda batch: collate_forward(batch, encoder.pad_id)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSimulatorModel(len(vocab), model_cfg).to(device)
    resumed_extra = {}
    if args.resume_checkpoint:
        resumed_extra = load_compatible_checkpoint(args.resume_checkpoint, model, vocab, device)
        print(f"resumed compatible model weights from {args.resume_checkpoint}", flush=True)
        if "best_val_possession_mae_pp" in resumed_extra:
            print(f"checkpoint best_val_possession_mae_pp={resumed_extra['best_val_possession_mae_pp']:.2f}", flush=True)
    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"using cuda devices={gpu_count}: {gpu_names}", flush=True)
        if gpu_count > 1:
            model = torch.nn.DataParallel(model)
            print("wrapped model with torch.nn.DataParallel", flush=True)
    else:
        print("using cpu", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scaler = GradScaler(enabled=train_cfg.amp and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_warmup_lr_lambda(epoch, args.warmup_epochs, train_cfg.epochs),
    )

    rollout_eval_files = []
    if args.rollout_eval_every_epochs > 0:
        holdout = load_json(args.pinned_holdout_json)
        rollout_eval_files = [
            str(Path(data_cfg.events_dir) / mid) for mid in holdout["match_ids"][: args.rollout_eval_matches]
        ]
        print(f"rollout-eval enabled every {args.rollout_eval_every_epochs} epochs on {len(rollout_eval_files)} pinned-holdout matches", flush=True)

    best_mae = float("inf") if args.reset_best_metric else float(resumed_extra.get("best_val_possession_mae_pp", float("inf")))
    for _ in range(1, args.start_epoch):
        scheduler.step()
    for epoch in range(args.start_epoch, train_cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, train_cfg, encoder.pad_id, device, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, scaler, train_cfg, encoder.pad_id, device, train=False)
        current_lr = optimizer.param_groups[0]["lr"]
        rollout_note = ""
        if args.rollout_eval_every_epochs > 0 and (epoch % args.rollout_eval_every_epochs == 0 or epoch == train_cfg.epochs):
            rollout_mae = rollout_holdout_mae(model, vocab, encoder, rollout_eval_files, data_cfg.lineups_dir, device, DecodeConfig())
            rollout_note = f" rollout_mae={rollout_mae:.2f}pp(n={len(rollout_eval_files)})"
        print(
            f"epoch={epoch:03d} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_pos_mae={val_metrics['possession_mae_pp']:.2f}pp"
            f"{rollout_note}"
        )
        scheduler.step()
        epoch_extra = {
            "token_cfg": token_cfg.__dict__,
            "data_cfg": data_cfg.__dict__,
            "best_val_possession_mae_pp": best_mae,
            "test_files": test_files,
            "epoch": epoch,
            "val_pos_mae_pp": val_metrics["possession_mae_pp"],
        }
        epoch_ckpt_dir = out_dir / "epoch_checkpoints"
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            str(epoch_ckpt_dir / f"epoch{epoch:03d}.pt"),
            model.module if isinstance(model, torch.nn.DataParallel) else model,
            vocab.to_dict(),
            model_cfg,
            epoch_extra,
        )
        if val_metrics["possession_mae_pp"] < best_mae:
            best_mae = val_metrics["possession_mae_pp"]
            epoch_extra["best_val_possession_mae_pp"] = best_mae
            save_checkpoint(
                str(out_dir / "best_forward_simulator.pt"),
                model.module if isinstance(model, torch.nn.DataParallel) else model,
                vocab.to_dict(),
                model_cfg,
                epoch_extra,
            )
            print(f"saved best checkpoint with val_pos_mae={best_mae:.2f}pp")


if __name__ == "__main__":
    main()
