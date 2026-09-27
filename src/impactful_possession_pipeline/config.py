from dataclasses import dataclass
from typing import Optional, Tuple


SUB_ROLE_ACTIONS: Tuple[str, ...] = (
    "GOALKEEPER",
    "DEFENDER",
    "WINGBACK",
    "DEFENSIVE_MIDFIELDER",
    "CENTRAL_MIDFIELDER",
    "ATTACKING_MIDFIELDER",
    "WINGER",
    "FORWARD",
)

TACTIC_FORMATION_ACTIONS: Tuple[str, ...] = (
    "4231",
    "433",
    "442",
    "4141",
    "4411",
    "352",
    "41212",
    "3421",
    "343",
)


def rich_action_tokens() -> Tuple[str, ...]:
    tokens = []
    for team in ("HOME", "AWAY"):
        for role in SUB_ROLE_ACTIONS:
            tokens.append(f"ACTION::{team}_SUB_ROLE_{role}")
        for formation in TACTIC_FORMATION_ACTIONS:
            tokens.append(f"ACTION::{team}_TACTIC_FORM_{formation}")
    return tuple(tokens)


FORMATION_STATE_VALUES: Tuple[str, ...] = TACTIC_FORMATION_ACTIONS + ("UNKNOWN",)


def formation_state_tokens() -> Tuple[str, ...]:
    """Separate namespace from ACTION::*_TACTIC_FORM_* so 'formation is currently
    X' (passive, re-announced periodically) is never confused with 'just switched
    to X' (a discrete event/intervention). Grammar-forced during decoding, never
    sampled -- see decode.py."""
    tokens = []
    for team in ("HOME", "AWAY"):
        for formation in FORMATION_STATE_VALUES:
            tokens.append(f"STATE::{team}_FORM_{formation}")
    return tuple(tokens)


@dataclass
class DataConfig:
    events_dir: str = "/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/events"
    lineups_dir: str = "/mnt/walkure_public/research/footballai/datasets/statsbomb/raw/open-data-master/data/lineups"
    max_matches: int = 3400
    train_fraction: float = 0.82
    val_fraction: float = 0.08
    seed: int = 42
    cut_minutes: Tuple[int, ...] = (30, 35, 40, 45, 50, 55, 60, 65, 70)
    max_examples_per_match: int = 9
    action_anchor_pairs: bool = True


@dataclass
class TokenConfig:
    x_bins: int = 12
    y_bins: int = 8
    max_dt_minutes: int = 5
    max_player_role_slots: int = 8
    score_diff_min: int = -4
    score_diff_max: int = 4
    action_tokens: Tuple[str, ...] = (
        "ACTION::NOOP",
        "ACTION::HOME_SUB_ATTACK",
        "ACTION::HOME_SUB_NEUTRAL",
        "ACTION::HOME_SUB_DEFEND",
        "ACTION::HOME_TACTIC_ATTACK",
        "ACTION::HOME_TACTIC_NEUTRAL",
        "ACTION::HOME_TACTIC_DEFEND",
        "ACTION::AWAY_SUB_ATTACK",
        "ACTION::AWAY_SUB_NEUTRAL",
        "ACTION::AWAY_SUB_DEFEND",
        "ACTION::AWAY_TACTIC_ATTACK",
        "ACTION::AWAY_TACTIC_NEUTRAL",
        "ACTION::AWAY_TACTIC_DEFEND",
    )
    rich_actions: bool = False
    emit_formation_state: bool = False
    formation_state_interval_minutes: int = 15


def action_tokens_for_config(cfg: TokenConfig) -> Tuple[str, ...]:
    if not cfg.rich_actions:
        return cfg.action_tokens
    return tuple(dict.fromkeys(tuple(cfg.action_tokens) + rich_action_tokens()))


@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    max_len: int = 4096
    local_attention_window: int = 768
    value_pool_window: Optional[int] = None


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 2
    amp: bool = True
    possession_weight: float = 2.0
    count_weight: float = 0.15
    uncertainty_weight: float = 0.05
    event_token_weight: float = 1.0
    team_token_weight: float = 1.0
    role_token_weight: float = 1.0
    location_token_weight: float = 1.0
    possession_flip_token_weight: float = 1.0
    goal_token_weight: float = 1.0
    save_every_epochs: int = 1


@dataclass
class DecodeConfig:
    top_p: float = 0.90
    temperature: float = 0.80
    prompt_tail_tokens: int = 1800
    min_events_per_minute: int = 33
    max_events_per_minute: int = 49
    target_events_per_minute: Optional[float] = None
    event_density_tolerance: int = 2
    target_home_possession: Optional[float] = None
    external_possession_weight: float = 0.0
    possession_guidance_prior_events: float = 0.0
    include_organic_actions: bool = True
    possession_guidance_strength: float = 4.0
    enforce_possession_state: bool = False
    event_type_prior: Optional[dict] = None
    role_prior: Optional[dict] = None
    lx_prior: Optional[dict] = None
    ly_prior: Optional[dict] = None
    prior_bias_strength: float = 0.0
    conditional_prior: Optional[dict] = None
    conditional_prior_bias_strength: float = 0.0
    football_structure_constraints: bool = False
    pass_receipt_boost: float = 0.0
    pass_carry_boost: float = 0.0
    overproduced_event_penalty: float = 0.0
    max_pass_run: int = 4
    force_receipt_after_pass_run: int = 3


@dataclass
class PlannerConfig:
    step_minutes: int = 5
    rollout_seeds: int = 12
    max_subs: int = 3
    max_tactical_shifts: int = 3
    tactical_cooldown_steps: int = 1
    min_delta_gate: float = 0.010
    intervention_cost: float = 0.004
    risk_penalty: float = 0.35
    uncertainty_penalty: float = 0.15
    allow_defensive_actions: bool = False
    # Search (fast, every candidate/minute) is value-head-only. Confirmation
    # (slow, generation-based rollouts) only runs on a short list of top
    # value-head candidates, and only when explicitly requested -- never as
    # the primary search mechanism. See planner.py.
    confirm_top_k: int = 2
    confirm_rollout_seeds: int = 20
    confirm_lcb_alpha: float = 0.10
    confirm_min_lcb: float = 0.0
    confirm_min_positive_seed_rate: float = 0.0
    confirm_horizon: str = "step"
    final_confirm_horizon: str = "none"
    final_confirm_rollout_seeds: int = 0
    final_confirm_min_lcb: float = 0.0
    final_confirm_min_positive_seed_rate: float = 0.0
    final_confirm_all_shortlist: bool = False


@dataclass
class MCTSConfig:
    """PUCT-guided root search over tactical actions at each decision point.

    Every candidate's value estimate comes from an actual generation rollout (never
    the deterministic value head), so this sidesteps the value-head pooling-dilution
    issue entirely -- at the cost of one rollout per simulation, which is far more
    expensive per decision than planner.py's cheap value-head screen.
    """
    num_simulations: int = 10
    c_puct: float = 6.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    uniform_prior_weight: float = 0.05
    selection_policy: str = "visits"
    risk_penalty: float = 0.0
    noop_margin: float = 0.0
    reward_mode: str = "possession"
    shots_weight: float = 0.0
    possession_objective_weight: float = 1.0
