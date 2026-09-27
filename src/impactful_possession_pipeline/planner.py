from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from .config import DecodeConfig, PlannerConfig, TokenConfig, action_tokens_for_config
from .decode import decode_frames, generate_continuation
from .model import ForwardSimulatorModel
from .utils import bootstrap_mean_lcb, home_possession_ratio, mean, std
from .vocabulary import Vocab


HOME_ACTIONS = [
    "ACTION::HOME_SUB_ATTACK",
    "ACTION::HOME_SUB_NEUTRAL",
    "ACTION::HOME_SUB_DEFEND",
    "ACTION::HOME_TACTIC_ATTACK",
    "ACTION::HOME_TACTIC_NEUTRAL",
    "ACTION::HOME_TACTIC_DEFEND",
]


@dataclass
class ActionScore:
    """Primary (cheap, deterministic) value-head score for one candidate action.

    This is what search/ranking uses. It does NOT involve any generation or
    seeds -- `predict_values` is a single deterministic forward pass, so the
    2026-07-18/19 "same candidate swings +90pp to -90pp across seeds" failure
    mode cannot happen here by construction. It CAN still be wrong in ways a
    deterministic model is wrong (miscalibration, confounded training
    signal) -- that's what `ConfirmationResult` / full-holdout screening is
    for, not seed noise.
    """
    action: str
    noop_value: float
    action_value: float
    delta: float
    uncertainty: float
    utility: float


@dataclass
class ConfirmationResult:
    """Slow, generation-based rollout confirmation for a shortlisted candidate.

    Only ever run on a short list of top ActionScore candidates, with a large
    seed count, and gated on a bootstrap lower confidence bound -- not a
    single match's seed spread -- before anything gets called a finding.
    """
    action: str
    seeds: int
    mean_delta: float
    std_delta: float
    lcb: float
    positive_seed_rate: float
    deltas: List[float] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return self.lcb > 0.0


def action_kind(action: str) -> str:
    if "_SUB_" in action:
        return "sub"
    if "_TACTIC_" in action:
        return "tactic"
    return "noop"


def home_actions(token_cfg: TokenConfig) -> list[str]:
    return [token for token in action_tokens_for_config(token_cfg) if token.startswith("ACTION::HOME_")]


def legal_actions(
    vocab: Vocab,
    used_subs: int,
    used_tactics: int,
    cooldown: int,
    cfg: PlannerConfig,
    token_cfg: TokenConfig,
    allowed_actions: Optional[Sequence[str]] = None,
) -> List[str]:
    actions = ["ACTION::NOOP"]
    allowed = set(allowed_actions) if allowed_actions else None
    for action in home_actions(token_cfg):
        if allowed is not None and action not in allowed:
            continue
        if not cfg.allow_defensive_actions and action.endswith("_DEFEND"):
            continue
        if action_kind(action) == "sub" and used_subs >= cfg.max_subs:
            continue
        if action_kind(action) == "tactic" and (used_tactics >= cfg.max_tactical_shifts or cooldown > 0):
            continue
        if action in vocab.stoi:
            actions.append(action)
    return actions


def value_head_estimate(model: ForwardSimulatorModel, ids: Sequence[int], device: torch.device) -> Tuple[float, float]:
    context = torch.tensor([list(ids)], dtype=torch.long, device=device)
    mask = torch.ones_like(context, dtype=torch.bool)
    pred = model.predict_values(context, mask)
    rest_pos = float(pred["possession"][0, 2].detach().cpu())
    rest_unc = float(pred["uncertainty"][0, 2].detach().cpu())
    return rest_pos, rest_unc


def score_action(
    model: ForwardSimulatorModel,
    vocab: Vocab,
    prompt_ids: Sequence[int],
    action: str,
    planner_cfg: PlannerConfig,
    device: torch.device,
) -> ActionScore:
    """Rank a candidate action using only the deterministic value head."""
    noop_value, noop_unc = value_head_estimate(model, prompt_ids, device)
    if action == "ACTION::NOOP":
        return ActionScore(action, noop_value, noop_value, 0.0, noop_unc, 0.0)
    action_context = list(prompt_ids) + [vocab.id(action)]
    action_value, action_unc = value_head_estimate(model, action_context, device)
    delta = action_value - noop_value
    uncertainty = max(noop_unc, action_unc)
    utility = delta - planner_cfg.risk_penalty * uncertainty - planner_cfg.intervention_cost
    return ActionScore(action, noop_value, action_value, delta, uncertainty, utility)


def rollout_value(
    model: ForwardSimulatorModel,
    vocab: Vocab,
    prompt_ids: Sequence[int],
    current_minute: int,
    action: Optional[str],
    token_cfg: TokenConfig,
    decode_cfg: DecodeConfig,
    planner_cfg: PlannerConfig,
    device: torch.device,
    seed: int,
    horizon: Optional[str] = None,
) -> float:
    action_id = None if action in (None, "ACTION::NOOP") else vocab.id(action)
    active_horizon = horizon or planner_cfg.confirm_horizon
    rollout_end_minute = 90 if active_horizon == "rest" else min(90, current_minute + planner_cfg.step_minutes)
    generated = generate_continuation(
        model,
        vocab,
        prompt_ids,
        start_minute=current_minute,
        end_minute=rollout_end_minute,
        token_cfg=token_cfg,
        decode_cfg=decode_cfg,
        device=device,
        seed=seed,
        injected_action_id=action_id,
    )
    frames = decode_frames(vocab, generated, start_minute=current_minute)
    return home_possession_ratio(frames)


def confirm_action(
    model: ForwardSimulatorModel,
    vocab: Vocab,
    prompt_ids: Sequence[int],
    current_minute: int,
    action: str,
    token_cfg: TokenConfig,
    decode_cfg: DecodeConfig,
    planner_cfg: PlannerConfig,
    device: torch.device,
    base_seed: int,
    num_seeds: Optional[int] = None,
    horizon: Optional[str] = None,
) -> ConfirmationResult:
    """Large-seed paired no-op-vs-action rollout confirmation for ONE shortlisted candidate.

    This is intentionally the only place noisy free-running generation feeds
    into a decision, and only ever for a short list of already value-head-
    ranked candidates -- not the search mechanism itself.
    """
    seeds = num_seeds or planner_cfg.confirm_rollout_seeds
    deltas: List[float] = []
    for i in range(seeds):
        seed = base_seed + i
        noop_v = rollout_value(model, vocab, prompt_ids, current_minute, None, token_cfg, decode_cfg, planner_cfg, device, seed, horizon=horizon)
        action_v = rollout_value(model, vocab, prompt_ids, current_minute, action, token_cfg, decode_cfg, planner_cfg, device, seed, horizon=horizon)
        deltas.append(action_v - noop_v)
    lcb = bootstrap_mean_lcb(deltas, alpha=planner_cfg.confirm_lcb_alpha, seed=base_seed)
    positive_rate = mean([1.0 if d > 0 else 0.0 for d in deltas])
    return ConfirmationResult(
        action=action,
        seeds=seeds,
        mean_delta=mean(deltas),
        std_delta=std(deltas),
        lcb=lcb,
        positive_seed_rate=positive_rate,
        deltas=deltas,
    )


def choose_action(
    model: ForwardSimulatorModel,
    vocab: Vocab,
    prompt_ids: Sequence[int],
    current_minute: int,
    used_subs: int,
    used_tactics: int,
    cooldown: int,
    token_cfg: TokenConfig,
    decode_cfg: DecodeConfig,
    planner_cfg: PlannerConfig,
    device: torch.device,
    seed: int,
    allowed_actions: Optional[Sequence[str]] = None,
    require_confirmation: bool = False,
) -> Tuple[str, List[ActionScore], List[ConfirmationResult]]:
    candidates = legal_actions(
        vocab, used_subs, used_tactics, cooldown, planner_cfg, token_cfg, allowed_actions
    )
    scores = [
        score_action(model, vocab, prompt_ids, action, planner_cfg, device)
        for action in candidates
    ]
    scores.sort(key=lambda s: s.utility, reverse=True)

    confirmations: List[ConfirmationResult] = []
    if not require_confirmation:
        best = scores[0]
        if best.action == "ACTION::NOOP" or best.delta < planner_cfg.min_delta_gate or best.utility <= 0:
            return "ACTION::NOOP", scores, confirmations
        return best.action, scores, confirmations

    # Confirmation mode: only rollout-generate for the top-K non-noop value-head
    # candidates, and only accept one whose bootstrap LCB clears zero.
    shortlist = [s for s in scores if s.action != "ACTION::NOOP" and s.delta >= planner_cfg.min_delta_gate and s.utility > 0]
    shortlist = shortlist[: planner_cfg.confirm_top_k]
    for idx, candidate in enumerate(shortlist):
        result = confirm_action(
            model, vocab, prompt_ids, current_minute, candidate.action,
            token_cfg, decode_cfg, planner_cfg, device, seed + 1000 * idx,
        )
        confirmations.append(result)
        step_passes = (
            result.lcb >= planner_cfg.confirm_min_lcb
            and result.positive_seed_rate >= planner_cfg.confirm_min_positive_seed_rate
        )
        should_final_confirm = planner_cfg.final_confirm_horizon != "none" and (step_passes or planner_cfg.final_confirm_all_shortlist)
        if should_final_confirm:
            final_result = confirm_action(
                model, vocab, prompt_ids, current_minute, candidate.action,
                token_cfg, decode_cfg, planner_cfg, device,
                seed + 50_000 + 1000 * idx,
                num_seeds=planner_cfg.final_confirm_rollout_seeds or planner_cfg.confirm_rollout_seeds,
                horizon=planner_cfg.final_confirm_horizon,
            )
            confirmations.append(final_result)
            if (
                final_result.lcb >= planner_cfg.final_confirm_min_lcb
                and final_result.positive_seed_rate >= planner_cfg.final_confirm_min_positive_seed_rate
            ):
                return candidate.action, scores, confirmations
            continue
        if step_passes:
            return candidate.action, scores, confirmations
    return "ACTION::NOOP", scores, confirmations


def plan_second_half(
    model: ForwardSimulatorModel,
    vocab: Vocab,
    prompt_ids: Sequence[int],
    token_cfg: TokenConfig,
    decode_cfg: DecodeConfig,
    planner_cfg: PlannerConfig,
    device: torch.device,
    seed: int = 42,
    end_minute: int = 90,
    cut_minute: int = 45,
    allowed_actions: Optional[Sequence[str]] = None,
    require_confirmation: bool = False,
) -> dict:
    ctx = list(prompt_ids)
    interventions = []
    used_subs = 0
    used_tactics = 0
    cooldown = 0
    for minute in range(cut_minute, end_minute, planner_cfg.step_minutes):
        action, scores, confirmations = choose_action(
            model, vocab, ctx, minute, used_subs, used_tactics, cooldown,
            token_cfg, decode_cfg, planner_cfg, device, seed + minute * 100,
            allowed_actions=allowed_actions, require_confirmation=require_confirmation,
        )
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
            token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=seed + minute
        )
        eos_id = vocab.id("<eos>")
        ctx.extend([tok for tok in generated if tok != eos_id])
        interventions.append({
            "minute": minute,
            "chosen_action": action,
            "scores": [s.__dict__ for s in scores],
            "confirmations": [c.__dict__ for c in confirmations],
        })
    frames = decode_frames(vocab, ctx, start_minute=0)
    rest_frames = [frame for frame in frames if cut_minute <= frame.get("minute", 0) < end_minute]
    return {
        "interventions": interventions,
        "simulated_home_possession": home_possession_ratio(rest_frames),
        "full_match_home_possession": home_possession_ratio(frames),
        "used_subs": used_subs,
        "used_tactical_shifts": used_tactics,
        "allowed_actions": list(allowed_actions) if allowed_actions else None,
    }
