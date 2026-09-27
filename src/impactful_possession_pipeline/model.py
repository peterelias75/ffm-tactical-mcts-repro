from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .dataset import IGNORE_INDEX


class CausalBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = int(cfg.d_model * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = ~attention_mask
        h = self.ln1(x)
        y, _ = self.attn(h, h, h, attn_mask=causal, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class ForwardSimulatorModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([CausalBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.value_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 3),
        )
        self.count_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 2),
        )
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 3),
        )

    def trim_to_max_len(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.size(1) > self.cfg.max_len:
            input_ids = input_ids[:, -self.cfg.max_len:]
            attention_mask = attention_mask[:, -self.cfg.max_len:]
        return input_ids, attention_mask

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = self.trim_to_max_len(input_ids, attention_mask)
        pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.ln_f(x)

    def pooled_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = self.trim_to_max_len(input_ids, attention_mask)
        h = self.encode(input_ids, attention_mask)
        mask = attention_mask
        window = self.cfg.value_pool_window
        if window:
            # Context is right-padded (valid tokens occupy the first valid_length
            # positions per example), so "last window valid tokens" differs per
            # example in a batch -- a plain h[:, -window:] slice would silently
            # grab trailing padding for any shorter-than-max example. Compute the
            # per-example recency window explicitly instead.
            valid_lengths = attention_mask.sum(dim=1, keepdim=True)
            positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
            mask = attention_mask & (positions >= (valid_lengths - window))
        weights = mask.float().unsqueeze(-1)
        return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, context_ids: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = self.trim_to_max_len(input_ids, attention_mask)
        h = self.encode(input_ids, attention_mask)
        logits = self.lm_head(h)
        if context_ids is None:
            pooled = (h * attention_mask.float().unsqueeze(-1)).sum(dim=1) / attention_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            pooled = self.pooled_context(context_ids, context_mask)
        return {
            "logits": logits,
            "possession_logits": self.value_head(pooled),
            "count_logs": self.count_head(pooled),
            "uncertainty_logs": self.uncertainty_head(pooled),
        }

    @torch.no_grad()
    def next_token_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = self.trim_to_max_len(input_ids, attention_mask)
        h = self.encode(input_ids, attention_mask)
        return self.lm_head(h[:, -1, :])

    @torch.no_grad()
    def predict_values(self, context_ids: torch.Tensor, context_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pooled_context(context_ids, context_mask)
        return {
            "possession": torch.sigmoid(self.value_head(pooled)),
            "counts": torch.expm1(self.count_head(pooled)).clamp_min(0.0),
            "uncertainty": torch.exp(self.uncertainty_head(pooled)).clamp_min(1e-4),
        }


def compute_losses(outputs: Dict[str, torch.Tensor], batch: dict, pad_id: int, possession_weight: float, count_weight: float, uncertainty_weight: float) -> Dict[str, torch.Tensor]:
    vocab_size = outputs["logits"].size(-1)
    flat_logits = outputs["logits"].reshape(-1, vocab_size)
    flat_labels = batch["labels"].reshape(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX, reduction="none")
    valid = flat_labels != IGNORE_INDEX
    if "label_weights" in batch:
        flat_weights = batch["label_weights"].reshape(-1).to(per_token_loss.device)
        token_loss = (per_token_loss * flat_weights * valid.float()).sum() / (flat_weights * valid.float()).sum().clamp_min(1.0)
    else:
        token_loss = per_token_loss[valid].mean() if valid.any() else per_token_loss.mean()
    targets = torch.stack([batch["target_pos_5"], batch["target_pos_10"], batch["target_pos_rest"]], dim=1)
    pred_pos = torch.sigmoid(outputs["possession_logits"])
    abs_err = (pred_pos - targets).abs()
    possession_loss = abs_err.mean()
    counts = torch.stack([batch["target_count_5"], batch["target_count_10"]], dim=1)
    count_loss = F.smooth_l1_loss(outputs["count_logs"], torch.log1p(counts))
    uncertainty_target = torch.log(abs_err.detach().clamp_min(1e-4))
    uncertainty_loss = F.smooth_l1_loss(outputs["uncertainty_logs"], uncertainty_target)
    total = token_loss + possession_weight * possession_loss + count_weight * count_loss + uncertainty_weight * uncertainty_loss
    return {
        "loss": total,
        "token_loss": token_loss.detach(),
        "possession_mae_pp": (abs_err.mean() * 100.0).detach(),
        "count_loss": count_loss.detach(),
        "uncertainty_loss": uncertainty_loss.detach(),
    }


def save_checkpoint(path: str, model: ForwardSimulatorModel, vocab_obj: dict, cfg: ModelConfig, extra: Optional[dict] = None) -> None:
    torch.save({"model": model.state_dict(), "vocab": vocab_obj, "model_cfg": asdict(cfg), "extra": extra or {}}, path)


def load_checkpoint(path: str, device: torch.device) -> Tuple[ForwardSimulatorModel, dict, ModelConfig, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = ForwardSimulatorModel(len(ckpt["vocab"]["itos"]), cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt["vocab"], cfg, ckpt.get("extra", {})
