"""PUCT-guided Monte Carlo Tree Search over tactical interventions.

Implements the tactical-search method described in the project paper: at each
decision point, blend the forward model's own action-token probabilities with a
small uniform prior and Dirichlet exploration noise at the root, then allocate a
fixed simulation budget across candidate actions using the PUCT score

    f_PUCT = Q + c_puct * P * sqrt(N_total) / (1 + n),

where Q is the running mean possession reward of an action's rollouts so far, P is
its (noise-mixed) prior, n is its visit count, and N_total is the parent's total
visit count. Each simulation is one free-running generation rollout of the chosen
action (never a deterministic value-head point estimate), so this sidesteps the
value-head pooling-dilution issue entirely -- at the cost of being far more
expensive per decision than planner.py's cheap value-head screen. After the
simulation budget is spent, the action with the highest visit count is taken as
the real decision, matching standard (AlphaZero-style) root-level action selection.

This is deliberately a single-level (root-only) search per decision point, not a
multi-ply game tree: "leaves" are the immediate candidate actions themselves, each
evaluated by a rollout, repeated for num_simulations budget, then the match is
advanced by generation under the chosen action before moving to the next decision
point 5 minutes later.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import DecodeConfig, MCTSConfig, PlannerConfig, TokenConfig
from .decode import decode_frames, generate_continuation
from .model import ForwardSimulatorModel
from .planner import action_kind, legal_actions
from .utils import home_possession_ratio, home_shot_count, home_shot_ratio
from .vocabulary import Vocab


@dataclass
class ActionNode:
    action: str
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    values: List[float] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    @property
    def std_value(self) -> float:
        if self.visit_count <= 1:
            return 0.0
        return float(np.std(self.values))

    def conservative_value(self, risk_penalty: float) -> float:
        return self.mean_value - risk_penalty * self.std_value


def action_priors(
    model: ForwardSimulatorModel,
    vocab: Vocab,
    prompt_ids: Sequence[int],
    candidates: Sequence[str],
    device: torch.device,
) -> dict:
    """P(action) from the model's own next-token distribution, restricted to the
    legal action tokens at this decision point -- this is the "learned prior" the
    paper blends with uniform weight and Dirichlet noise, not a separately trained
    policy network."""
    context = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    mask = torch.ones_like(context, dtype=torch.bool)
    logits = model.next_token_logits(context, mask)[0]
    action_ids = [vocab.id(a) for a in candidates]
    action_logits = logits[action_ids]
    probs = torch.softmax(action_logits.float(), dim=-1).detach().cpu().numpy()
    return {a: float(p) for a, p in zip(candidates, probs)}


def mcts_decide(
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
    mcts_cfg: MCTSConfig,
    device: torch.device,
    seed: int,
    allowed_actions: Optional[Sequence[str]] = None,
    lookahead_minutes: Optional[int] = None,
    rollout_max_new_tokens: Optional[int] = None,
) -> Tuple[str, List[ActionNode]]:
    """Run one root-level PUCT search over legal actions at this decision point.

    `rollout_max_new_tokens` is the literal generated-token budget used to
    *evaluate* each candidate. It is deliberately independent of
    `planner_cfg.step_minutes`, which controls real match advancement.

    `lookahead_minutes` is retained for compatibility with older callers and
    only limits the rollout time horizon. New reproduction runs use a
    90-minute time ceiling plus the explicit token budget.
    """
    candidates = legal_actions(vocab, used_subs, used_tactics, cooldown, planner_cfg, token_cfg, allowed_actions)
    raw_priors = action_priors(model, vocab, prompt_ids, candidates, device)

    rng = np.random.default_rng(seed)
    total_prior = sum(raw_priors.values()) or 1.0
    uniform = 1.0 / len(candidates)
    noise = rng.dirichlet([mcts_cfg.dirichlet_alpha] * len(candidates))
    nodes = []
    for i, action in enumerate(candidates):
        p = raw_priors[action] / total_prior
        p = (1 - mcts_cfg.uniform_prior_weight) * p + mcts_cfg.uniform_prior_weight * uniform
        p = (1 - mcts_cfg.dirichlet_epsilon) * p + mcts_cfg.dirichlet_epsilon * noise[i]
        nodes.append(ActionNode(action=action, prior=p))

    horizon = (90 - current_minute) if rollout_max_new_tokens is not None else (
        planner_cfg.step_minutes if lookahead_minutes is None else lookahead_minutes
    )

    for sim in range(mcts_cfg.num_simulations):
        total_n = sum(n.visit_count for n in nodes)

        def puct(node: ActionNode) -> float:
            return node.mean_value + mcts_cfg.c_puct * node.prior * (total_n ** 0.5) / (1 + node.visit_count)

        chosen = max(nodes, key=puct)
        rollout_seed = seed + sim * 7919

        def rollout_value(action: str) -> float:
            action_id = None if action == "ACTION::NOOP" else vocab.id(action)
            if rollout_max_new_tokens == 0 or horizon <= 0:
                # No-lookahead: judge candidates on the injected action token
                # alone, with no generated continuation at all.
                frames = decode_frames(vocab, [action_id] if action_id is not None else [], start_minute=current_minute)
                return (
                    mcts_cfg.shots_weight * home_shot_ratio(frames)
                    + mcts_cfg.possession_objective_weight * home_possession_ratio(frames)
                )
            generated = generate_continuation(
                model, vocab, prompt_ids, current_minute, min(90, current_minute + horizon),
                token_cfg=token_cfg, decode_cfg=decode_cfg, device=device, seed=rollout_seed,
                injected_action_id=action_id,
                max_new_tokens=rollout_max_new_tokens,
            )
            frames = decode_frames(vocab, generated, start_minute=current_minute)
            return (
                mcts_cfg.shots_weight * home_shot_ratio(frames)
                + mcts_cfg.possession_objective_weight * home_possession_ratio(frames)
            )

        if mcts_cfg.reward_mode == "paired_delta":
            reward = 0.0 if chosen.action == "ACTION::NOOP" else rollout_value(chosen.action) - rollout_value("ACTION::NOOP")
        else:
            reward = rollout_value(chosen.action)
        chosen.visit_count += 1
        chosen.value_sum += reward
        chosen.values.append(reward)

    if mcts_cfg.reward_mode == "paired_delta":
        noop = next((n for n in nodes if n.action == "ACTION::NOOP"), None)
        best = max(nodes, key=lambda n: n.conservative_value(mcts_cfg.risk_penalty))
        if best.action == "ACTION::NOOP" or best.conservative_value(mcts_cfg.risk_penalty) < mcts_cfg.noop_margin:
            best = noop or best
    elif mcts_cfg.selection_policy == "risk_adjusted":
        noop = next((n for n in nodes if n.action == "ACTION::NOOP"), None)
        noop_floor = noop.conservative_value(mcts_cfg.risk_penalty) if noop else 0.0
        best = max(nodes, key=lambda n: n.conservative_value(mcts_cfg.risk_penalty))
        if best.action != "ACTION::NOOP" and best.conservative_value(mcts_cfg.risk_penalty) < noop_floor + mcts_cfg.noop_margin:
            best = noop or best
    elif mcts_cfg.selection_policy == "mean":
        best = max(nodes, key=lambda n: n.mean_value)
    else:
        best = max(nodes, key=lambda n: n.visit_count)
    return best.action, nodes


def plan_second_half_mcts(
    model: ForwardSimulatorModel,
    vocab: Vocab,
    prompt_ids: Sequence[int],
    token_cfg: TokenConfig,
    decode_cfg: DecodeConfig,
    planner_cfg: PlannerConfig,
    mcts_cfg: MCTSConfig,
    device: torch.device,
    seed: int = 42,
    end_minute: int = 90,
    cut_minute: int = 45,
    allowed_actions: Optional[Sequence[str]] = None,
    lookahead_minutes: Optional[int] = None,
    rollout_max_new_tokens: Optional[int] = None,
    intervention_cooldown_minutes: int = 1,
) -> dict:
    ctx = list(prompt_ids)
    interventions = []
    used_subs = 0
    used_tactics = 0
    cooldown = 0
    next_intervention_minute = cut_minute
    for minute in range(cut_minute, end_minute, planner_cfg.step_minutes):
        forced_noop = minute < next_intervention_minute
        if forced_noop:
            action, nodes = "ACTION::NOOP", []
        else:
            action, nodes = mcts_decide(
                model, vocab, ctx, minute, used_subs, used_tactics, cooldown,
                token_cfg, decode_cfg, planner_cfg, mcts_cfg, device,
                seed=seed + minute * 1000, allowed_actions=allowed_actions,
                lookahead_minutes=lookahead_minutes,
                rollout_max_new_tokens=rollout_max_new_tokens,
            )
        if action != "ACTION::NOOP":
            next_intervention_minute = minute + max(1, intervention_cooldown_minutes)
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
        interventions.append({
            "minute": minute,
            "chosen_action": action,
            "forced_by_intervention_cooldown": forced_noop,
            "nodes": [
                {"action": n.action, "prior": n.prior, "visit_count": n.visit_count, "mean_value": n.mean_value, "std_value": n.std_value, "conservative_value": n.conservative_value(mcts_cfg.risk_penalty)}
                for n in nodes
            ],
        })
    frames = decode_frames(vocab, ctx, start_minute=0)
    rest_frames = [f for f in frames if cut_minute <= f.get("minute", 0) < end_minute]
    return {
        "interventions": interventions,
        "simulated_home_possession": home_possession_ratio(rest_frames),
        "simulated_home_shots": home_shot_count(rest_frames),
        "used_subs": used_subs,
        "used_tactical_shifts": used_tactics,
    }
