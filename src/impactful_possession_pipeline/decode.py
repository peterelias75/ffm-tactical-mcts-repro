import math
from typing import List, Optional, Sequence

import torch

from .config import DecodeConfig, TokenConfig, action_tokens_for_config
from .model import ForwardSimulatorModel
from .utils import POSSESSION_EVENT_TYPES
from .vocabulary import EVENT_TYPES, FRAME_FIELDS, PLAYER_ROLES, Vocab

SAME_POSSESSION_EVENT_TYPES = {"Pass", "Carry", "Ball Receipt*", "Shot"}


def last_possession_team(vocab: Vocab, ids: Sequence[int]) -> Optional[str]:
    team = None
    for frame in decode_frames(vocab, ids, start_minute=0):
        if frame.get("event_type") in POSSESSION_EVENT_TYPES and frame.get("team") in {"home", "away"}:
            team = frame["team"]
    return team


def team_token_id(vocab: Vocab, team: str) -> int:
    return vocab.id(f"TEAM::{team}")


def last_formation_state(vocab: Vocab, ids: Sequence[int]) -> dict:
    state = {"home": "UNKNOWN", "away": "UNKNOWN"}
    for idx in ids:
        token = vocab.token(int(idx))
        if token.startswith("STATE::HOME_FORM_"):
            state["home"] = token.rsplit("_", 1)[-1]
        elif token.startswith("STATE::AWAY_FORM_"):
            state["away"] = token.rsplit("_", 1)[-1]
    return state


def formation_from_tactic_action(token: str) -> Optional[tuple]:
    """('home'|'away', formation_str) if token is an ACTION::*_TACTIC_FORM_* token, else None."""
    if "_TACTIC_FORM_" not in token:
        return None
    team = "home" if "::HOME_" in token else "away"
    formation = token.rsplit("_", 1)[-1]
    return team, formation


def apply_prior_bias(
    logits: torch.Tensor,
    vocab: Vocab,
    prefix: str,
    values: Sequence,
    prior: Optional[dict],
    strength: float,
) -> torch.Tensor:
    if not prior or strength <= 0:
        return logits
    token_ids = []
    log_probs = []
    for value in values:
        key = str(value)
        prob = float(prior.get(key, 0.0))
        token_ids.append(vocab.id(f"{prefix}::{value}"))
        log_probs.append(math.log(max(prob, 1e-9)))
    mean_log_prob = sum(log_probs) / max(1, len(log_probs))
    biased = logits.clone()
    for token_id, log_prob in zip(token_ids, log_probs):
        biased[token_id] += strength * (log_prob - mean_log_prob)
    return biased


def allowed_ids(vocab: Vocab, token_cfg: TokenConfig, stage: int, include_actions: bool) -> List[int]:
    if stage == 0 and include_actions:
        action_ids = [vocab.id(token) for token in action_tokens_for_config(token_cfg) if token != "ACTION::NOOP" and token in vocab.stoi]
        return action_ids + [vocab.id(f"EVT::{event_type}") for event_type in EVENT_TYPES]
    field = FRAME_FIELDS[stage]
    if field == "EVT":
        return [vocab.id(f"EVT::{event_type}") for event_type in EVENT_TYPES]
    if field == "TEAM":
        return [vocab.id("TEAM::home"), vocab.id("TEAM::away")]
    if field == "ROLE":
        return [vocab.id(f"ROLE::{role}") for role in PLAYER_ROLES]
    if field == "LX":
        return [vocab.id(f"LX::{x}") for x in range(token_cfg.x_bins)]
    if field == "LY":
        return [vocab.id(f"LY::{y}") for y in range(token_cfg.y_bins)]
    if field == "DT":
        return [vocab.id(f"DT::{dt}") for dt in range(token_cfg.max_dt_minutes + 1)]
    if field == "SCORE":
        return [vocab.id(f"SCORE::{score}") for score in range(token_cfg.score_diff_min, token_cfg.score_diff_max + 1)]
    if field == "PFLIP":
        return [vocab.id("PFLIP::0"), vocab.id("PFLIP::1")]
    if field == "GOAL":
        return [vocab.id("GOAL::0"), vocab.id("GOAL::1")]
    return [vocab.id("<sep>")]


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float, allowed: Optional[Sequence[int]] = None) -> int:
    row = logits.float().clone()
    if allowed is not None:
        mask = torch.full_like(row, float("-inf"))
        mask[list(allowed)] = row[list(allowed)]
        row = mask
    row = row / max(temperature, 1e-6)
    probs = torch.softmax(row, dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return int(allowed[0]) if allowed else int(torch.argmax(logits).item())
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative - sorted_probs > top_p
    sorted_probs[remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
    pick = torch.multinomial(sorted_probs, 1)
    return int(sorted_idx[pick].item())


def decode_frames(vocab: Vocab, ids: Sequence[int], start_minute: int = 0) -> List[dict]:
    frames: List[dict] = []
    cur = {}
    minute = start_minute
    for idx in ids:
        token = vocab.token(int(idx))
        if token in {"<pad>", "<bos>"}:
            continue
        if token == "<eos>":
            break
        if token.startswith("ACTION::"):
            frames.append({"minute": minute, "event_type": "Action", "team": "home" if "::HOME_" in token else ("away" if "::AWAY_" in token else "neutral"), "action": token, "width": 1})
            cur = {}
            continue
        if token == "<sep>":
            if cur:
                cur.setdefault("minute", minute)
                frames.append(cur)
            cur = {}
            continue
        prefix, _, value = token.partition("::")
        if prefix == "EVT":
            cur["event_type"] = value
        elif prefix == "TEAM":
            cur["team"] = value
        elif prefix == "ROLE":
            cur["role"] = value
        elif prefix == "LX":
            cur["lx"] = int(value)
        elif prefix == "LY":
            cur["ly"] = int(value)
        elif prefix == "DT":
            minute += int(value)
            cur["dt"] = int(value)
            cur["minute"] = minute
        elif prefix == "SCORE":
            cur["score_diff"] = int(value)
        elif prefix == "PFLIP":
            cur["possession_flip"] = int(value)
        elif prefix == "GOAL":
            cur["goal"] = int(value)
    return frames


@torch.no_grad()
def generate_continuation(
    model: ForwardSimulatorModel,
    vocab: Vocab,
    prompt_ids: Sequence[int],
    start_minute: int,
    end_minute: int,
    token_cfg: TokenConfig = TokenConfig(),
    decode_cfg: DecodeConfig = DecodeConfig(),
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    injected_action_id: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> List[int]:
    if max_new_tokens is not None and max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative or None")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if device is None:
        device = next(model.parameters()).device
    ids = list(prompt_ids)
    if len(ids) > decode_cfg.prompt_tail_tokens:
        ids = [vocab.id("<bos>")] + ids[-(decode_cfg.prompt_tail_tokens - 1):]

    generated: List[int] = []
    emit_state = token_cfg.emit_formation_state
    current_formation = last_formation_state(vocab, prompt_ids) if emit_state else {}
    state_interval = max(1, token_cfg.formation_state_interval_minutes)
    last_state_minute = {"home": start_minute, "away": start_minute}

    def emit_formation_state(team: str, minute: int) -> None:
        if max_new_tokens is not None and len(generated) >= max_new_tokens:
            return
        token = f"STATE::{team.upper()}_FORM_{current_formation.get(team, 'UNKNOWN')}"
        token_id = vocab.id(token)
        ids.append(token_id)
        generated.append(token_id)
        last_state_minute[team] = minute

    if injected_action_id is not None:
        ids.append(int(injected_action_id))
        if emit_state:
            hit = formation_from_tactic_action(vocab.token(int(injected_action_id)))
            if hit is not None:
                current_formation[hit[0]] = hit[1]
    if emit_state:
        # Re-anchor both teams' formation right at the cut point, so a
        # possibly-truncated prompt never leaves the continuation without a
        # current reference (and so any injected intervention is reflected
        # immediately, matching how training data re-announces state right
        # after a real Tactical Shift).
        emit_formation_state("home", start_minute)
        emit_formation_state("away", start_minute)

    stage = 0
    minute = start_minute
    events_this_minute = 0
    pending_event_type: Optional[str] = None
    pending_team: Optional[str] = None
    prompt_frames = decode_frames(vocab, prompt_ids, start_minute=0)
    last_event_type = next((frame.get("event_type") for frame in reversed(prompt_frames) if frame.get("event_type") != "Action"), "<START>")
    current_possession_team = last_possession_team(vocab, prompt_ids)
    pass_run = 0
    pressure_run = 0
    recovery_run = 0
    possession_home_events = 0
    possession_total_events = 0
    target_home_possession: Optional[float] = None
    if decode_cfg.possession_guidance_strength > 0:
        context = torch.tensor([ids], dtype=torch.long, device=device)
        context_mask = torch.ones_like(context, dtype=torch.bool)
        target_home_possession = float(model.predict_values(context, context_mask)["possession"][0, 2].detach().cpu())
        if decode_cfg.target_home_possession is not None:
            external_weight = min(1.0, max(0.0, float(decode_cfg.external_possession_weight)))
            target_home_possession = (1.0 - external_weight) * target_home_possession + external_weight * float(decode_cfg.target_home_possession)
        if target_home_possession is not None and decode_cfg.possession_guidance_prior_events > 0:
            possession_total_events = float(decode_cfg.possession_guidance_prior_events)
            possession_home_events = possession_total_events * max(0.0, min(1.0, target_home_possession))
    min_events_per_minute = decode_cfg.min_events_per_minute
    max_events_per_minute = decode_cfg.max_events_per_minute
    if decode_cfg.target_events_per_minute is not None:
        target_events = max(1, int(round(decode_cfg.target_events_per_minute)))
        tolerance = max(0, int(decode_cfg.event_density_tolerance))
        min_events_per_minute = max(1, target_events - tolerance)
        max_events_per_minute = max(min_events_per_minute, target_events + tolerance)
    minute_based_max_tokens = max(1, end_minute - start_minute + 1) * max_events_per_minute * 10
    max_tokens = minute_based_max_tokens if max_new_tokens is None else min(minute_based_max_tokens, max_new_tokens)
    while minute < end_minute and len(generated) < max_tokens:
        model_ids = ids[-decode_cfg.prompt_tail_tokens:] if len(ids) > decode_cfg.prompt_tail_tokens else ids
        x = torch.tensor([model_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(x, dtype=torch.bool)
        logits = model.next_token_logits(x, mask)[0]
        allow = allowed_ids(vocab, token_cfg, stage, include_actions=decode_cfg.include_organic_actions)
        if FRAME_FIELDS[stage] == "DT":
            if events_this_minute < min_events_per_minute:
                allow = [vocab.id("DT::0")]
            elif events_this_minute >= max_events_per_minute:
                allow = [vocab.id("DT::1")]
            else:
                allow = [vocab.id("DT::0"), vocab.id("DT::1")]
        if (
            decode_cfg.enforce_possession_state
            and FRAME_FIELDS[stage] == "TEAM"
            and pending_event_type in SAME_POSSESSION_EVENT_TYPES
            and current_possession_team in {"home", "away"}
        ):
            allow = [team_token_id(vocab, current_possession_team)]
        if (
            decode_cfg.enforce_possession_state
            and FRAME_FIELDS[stage] == "PFLIP"
            and pending_event_type in POSSESSION_EVENT_TYPES
            and pending_team in {"home", "away"}
            and current_possession_team in {"home", "away"}
        ):
            flipped = int(pending_team != current_possession_team)
            allow = [vocab.id(f"PFLIP::{flipped}")]
        field = FRAME_FIELDS[stage]
        conditional_prior = decode_cfg.conditional_prior or {}
        conditional_strength = decode_cfg.conditional_prior_bias_strength
        if field == "EVT":
            logits = apply_prior_bias(logits, vocab, "EVT", EVENT_TYPES, decode_cfg.event_type_prior, decode_cfg.prior_bias_strength)
            event_priors = conditional_prior.get("event_type_given_prev", {})
            logits = apply_prior_bias(logits, vocab, "EVT", EVENT_TYPES, event_priors.get(str(last_event_type)), conditional_strength)
            if decode_cfg.football_structure_constraints:
                logits = logits.clone()
                pass_id = vocab.id("EVT::Pass")
                receipt_id = vocab.id("EVT::Ball Receipt*")
                carry_id = vocab.id("EVT::Carry")
                pressure_id = vocab.id("EVT::Pressure")
                recovery_id = vocab.id("EVT::Ball Recovery")
                if last_event_type == "Pass":
                    logits[receipt_id] += decode_cfg.pass_receipt_boost
                    logits[carry_id] += decode_cfg.pass_carry_boost
                    logits[pass_id] -= decode_cfg.overproduced_event_penalty
                if pass_run >= decode_cfg.force_receipt_after_pass_run:
                    allow = [receipt_id, carry_id]
                elif pass_run >= decode_cfg.max_pass_run:
                    allow = [tok for tok in allow if tok != pass_id]
                if decode_cfg.overproduced_event_penalty > 0:
                    logits[pressure_id] -= decode_cfg.overproduced_event_penalty
                    logits[recovery_id] -= 0.75 * decode_cfg.overproduced_event_penalty
                    if pressure_run >= 2:
                        logits[pressure_id] -= decode_cfg.overproduced_event_penalty
                    if recovery_run >= 1:
                        logits[recovery_id] -= decode_cfg.overproduced_event_penalty
        elif field == "ROLE":
            logits = apply_prior_bias(logits, vocab, "ROLE", PLAYER_ROLES, decode_cfg.role_prior, decode_cfg.prior_bias_strength)
            role_priors = conditional_prior.get("role_given_event", {})
            logits = apply_prior_bias(logits, vocab, "ROLE", PLAYER_ROLES, role_priors.get(str(pending_event_type)), conditional_strength)
        elif field == "LX":
            logits = apply_prior_bias(logits, vocab, "LX", range(token_cfg.x_bins), decode_cfg.lx_prior, decode_cfg.prior_bias_strength)
            lx_priors = conditional_prior.get("lx_given_event", {})
            logits = apply_prior_bias(logits, vocab, "LX", range(token_cfg.x_bins), lx_priors.get(str(pending_event_type)), conditional_strength)
        elif field == "LY":
            logits = apply_prior_bias(logits, vocab, "LY", range(token_cfg.y_bins), decode_cfg.ly_prior, decode_cfg.prior_bias_strength)
            ly_priors = conditional_prior.get("ly_given_event", {})
            logits = apply_prior_bias(logits, vocab, "LY", range(token_cfg.y_bins), ly_priors.get(str(pending_event_type)), conditional_strength)
        if FRAME_FIELDS[stage] == "TEAM" and pending_event_type in POSSESSION_EVENT_TYPES and target_home_possession is not None:
            current_home_share = possession_home_events / possession_total_events if possession_total_events else 0.5
            possession_error = target_home_possession - current_home_share
            team_bias = decode_cfg.possession_guidance_strength * possession_error
            logits = logits.clone()
            logits[vocab.id("TEAM::home")] += team_bias
            logits[vocab.id("TEAM::away")] -= team_bias
        next_id = top_p_sample(logits, decode_cfg.top_p, decode_cfg.temperature, allow)
        token = vocab.token(next_id)
        ids.append(next_id)
        generated.append(next_id)
        if token.startswith("ACTION::"):
            stage = 0
            pending_event_type = None
            pending_team = None
            if emit_state:
                hit = formation_from_tactic_action(token)
                if hit is not None:
                    current_formation[hit[0]] = hit[1]
                    emit_formation_state(hit[0], minute)
                for team in ("home", "away"):
                    if minute - last_state_minute[team] >= state_interval:
                        emit_formation_state(team, minute)
            continue
        if FRAME_FIELDS[stage] == "EVT":
            pending_event_type = token.split("::", 1)[1]
        elif FRAME_FIELDS[stage] == "TEAM":
            if token in {"TEAM::home", "TEAM::away"}:
                pending_team = token.split("::", 1)[1]
            if pending_event_type in POSSESSION_EVENT_TYPES:
                possession_total_events += 1
                if token == "TEAM::home":
                    possession_home_events += 1
        if FRAME_FIELDS[stage] == "DT":
            delta = int(token.split("::", 1)[1])
            minute += delta
            if delta > 0:
                events_this_minute = 0
        stage += 1
        if stage >= len(FRAME_FIELDS):
            if pending_event_type in POSSESSION_EVENT_TYPES and pending_team in {"home", "away"}:
                current_possession_team = pending_team
            if pending_event_type is not None:
                last_event_type = pending_event_type
                pass_run = pass_run + 1 if pending_event_type == "Pass" else 0
                pressure_run = pressure_run + 1 if pending_event_type == "Pressure" else 0
                recovery_run = recovery_run + 1 if pending_event_type == "Ball Recovery" else 0
            sep_id = vocab.id("<sep>")
            ids.append(sep_id)
            if len(generated) < max_tokens:
                generated.append(sep_id)
            events_this_minute += 1
            pending_event_type = None
            pending_team = None
            stage = 0
            if emit_state:
                for team in ("home", "away"):
                    if minute - last_state_minute[team] >= state_interval:
                        emit_formation_state(team, minute)
    generated.append(vocab.id("<eos>"))
    return generated
