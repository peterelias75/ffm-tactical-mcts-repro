import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import FORMATION_STATE_VALUES, TACTIC_FORMATION_ACTIONS, TokenConfig
from .utils import POSSESSION_EVENT_TYPES, clamp, load_json
from .vocabulary import EVENT_TYPES, PLAYER_ROLES, Vocab


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (name or "").lower()).strip()


def get_home_team_id(events: Sequence[dict]) -> Optional[int]:
    for event in events:
        if event.get("type", {}).get("name") == "Starting XI":
            return event.get("team", {}).get("id")
    for event in events:
        team_id = event.get("team", {}).get("id")
        if team_id is not None:
            return team_id
    return None


def role_from_position(position_name: str) -> str:
    p = (position_name or "").lower()
    if "goalkeeper" in p:
        return "goalkeeper"
    if "wing back" in p:
        return "wingback"
    if "back" in p or "defender" in p:
        return "defender"
    if "defensive midfield" in p:
        return "defensive_midfielder"
    if "attacking midfield" in p or "secondary striker" in p:
        return "attacking_midfielder"
    if "midfield" in p:
        return "central_midfielder"
    if "wing" in p:
        return "winger"
    if "forward" in p or "striker" in p:
        return "forward"
    return "central_midfielder"


def role_rank(role: str) -> int:
    return {
        "goalkeeper": 0,
        "defender": 1,
        "wingback": 2,
        "defensive_midfielder": 3,
        "central_midfielder": 4,
        "attacking_midfielder": 5,
        "winger": 6,
        "forward": 7,
    }.get(role, 4)


def load_lineup_roles(lineup_path: Optional[str]) -> Dict[str, str]:
    if not lineup_path or not os.path.exists(lineup_path):
        return {}
    lineup = load_json(lineup_path)
    roles: Dict[str, str] = {}
    for team in lineup:
        for player in team.get("lineup", []):
            name = normalize_name(player.get("player_name", ""))
            positions = player.get("positions", [])
            raw_position = positions[0].get("position", "") if positions else ""
            if name:
                roles[name] = role_from_position(raw_position)
    return roles


def replacement_role(event: dict, lineup_roles: Dict[str, str]) -> str:
    replacement = event.get("substitution", {}).get("replacement", {}).get("name", "")
    return lineup_roles.get(normalize_name(replacement), "central_midfielder")


def parse_formation(formation) -> Tuple[int, int, int]:
    if formation in (None, "Unknown", ""):
        return 4, 3, 3
    text = str(int(formation)) if isinstance(formation, (int, float)) else str(formation)
    parts = text.split("-") if "-" in text else list(text)
    nums = [int(p) for p in parts if str(p).isdigit()]
    if len(nums) == 3:
        return nums[0], nums[1], nums[2]
    if len(nums) == 4:
        return nums[0], nums[1] + nums[2], nums[3]
    if len(nums) >= 2:
        return nums[0], sum(nums[1:-1]) if len(nums) > 2 else 3, nums[-1]
    return 4, 3, 3


def action_for_substitution(event: dict, team: str, lineup_roles: Dict[str, str], cfg: TokenConfig | None = None) -> str:
    outgoing = role_from_position(event.get("position", {}).get("name", ""))
    incoming = replacement_role(event, lineup_roles)
    if cfg is not None and cfg.rich_actions:
        return f"ACTION::{team.upper()}_SUB_ROLE_{incoming.upper()}"
    if role_rank(incoming) > role_rank(outgoing):
        intent = "ATTACK"
    elif role_rank(incoming) < role_rank(outgoing):
        intent = "DEFEND"
    else:
        intent = "NEUTRAL"
    return f"ACTION::{team.upper()}_SUB_{intent}"


def action_for_tactical_shift(event: dict, team: str, previous_formation, cfg: TokenConfig | None = None) -> str:
    new_formation = event.get("tactics", {}).get("formation", previous_formation)
    if cfg is not None and cfg.rich_actions:
        formation = str(new_formation)
        if formation in TACTIC_FORMATION_ACTIONS:
            return f"ACTION::{team.upper()}_TACTIC_FORM_{formation}"
    old_d, _, old_a = parse_formation(previous_formation)
    new_d, _, new_a = parse_formation(new_formation)
    old_balance = old_a - old_d
    new_balance = new_a - new_d
    if new_balance > old_balance:
        intent = "ATTACK"
    elif new_balance < old_balance:
        intent = "DEFEND"
    else:
        intent = "NEUTRAL"
    return f"ACTION::{team.upper()}_TACTIC_{intent}"


def formation_state_value(formation) -> str:
    text = str(int(formation)) if isinstance(formation, (int, float)) else str(formation)
    return text if text in FORMATION_STATE_VALUES else "UNKNOWN"


def bin_location(location: Optional[list], cfg: TokenConfig) -> Tuple[int, int]:
    if not isinstance(location, list) or len(location) < 2:
        return 0, 0
    x = min(max(float(location[0]), 0.0), 120.0)
    y = min(max(float(location[1]), 0.0), 80.0)
    return min(int((x / 120.0) * cfg.x_bins), cfg.x_bins - 1), min(int((y / 80.0) * cfg.y_bins), cfg.y_bins - 1)


class StructuredStatsBombEncoder:
    def __init__(self, vocab: Vocab, cfg: TokenConfig = TokenConfig()):
        self.vocab = vocab
        self.cfg = cfg

    @property
    def pad_id(self) -> int:
        return self.vocab.id("<pad>")

    @property
    def bos_id(self) -> int:
        return self.vocab.id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.vocab.id("<eos>")

    @property
    def sep_id(self) -> int:
        return self.vocab.id("<sep>")

    def encode_match(self, events: Sequence[dict], lineup_path: Optional[str] = None) -> Tuple[List[int], List[dict]]:
        home_id = get_home_team_id(events)
        lineup_roles = load_lineup_roles(lineup_path)
        current_form = {"home": 433, "away": 433}
        possession_team = "home"
        score_home = 0
        score_away = 0
        previous_minute = 0
        ids = [self.bos_id]
        frames: List[dict] = []
        emit_state = self.cfg.emit_formation_state
        state_started = False
        last_state_minute = {"home": 0, "away": 0}
        interval = max(1, self.cfg.formation_state_interval_minutes)

        def emit_formation_state(team: str, minute: int) -> None:
            token = f"STATE::{team.upper()}_FORM_{formation_state_value(current_form.get(team))}"
            ids.append(self.vocab.id(token))
            frames.append({"minute": minute, "event_type": "State", "team": team, "action": token, "width": 1})
            last_state_minute[team] = minute

        ordered = sorted(events, key=lambda e: (e.get("period", 1), e.get("minute", 0), e.get("second", 0), e.get("index", 0)))
        for event in ordered:
            event_type = event.get("type", {}).get("name", "Pass")
            if event_type == "Starting XI":
                team = "home" if event.get("team", {}).get("id") == home_id else "away"
                current_form[team] = event.get("tactics", {}).get("formation", current_form[team])
                continue

            team = "home" if event.get("team", {}).get("id") == home_id else "away"
            minute = int(event.get("minute", previous_minute))
            dt = clamp(minute - previous_minute, 0, self.cfg.max_dt_minutes)
            previous_minute = minute

            if emit_state and not state_started:
                # Both Starting XI events have been seen by now (they sort first),
                # so current_form holds the true kickoff formations here.
                state_started = True
                emit_formation_state("home", minute)
                emit_formation_state("away", minute)

            if emit_state:
                for other_team in ("home", "away"):
                    if minute - last_state_minute[other_team] >= interval:
                        emit_formation_state(other_team, minute)

            if event_type == "Substitution":
                token = action_for_substitution(event, team, lineup_roles, self.cfg)
                ids.append(self.vocab.id(token))
                frames.append({"minute": minute, "event_type": "Action", "team": team, "action": token, "width": 1})
                continue

            if event_type == "Tactical Shift":
                token = action_for_tactical_shift(event, team, current_form.get(team), self.cfg)
                current_form[team] = event.get("tactics", {}).get("formation", current_form[team])
                ids.append(self.vocab.id(token))
                frames.append({"minute": minute, "event_type": "Action", "team": team, "action": token, "width": 1})
                if emit_state:
                    emit_formation_state(team, minute)
                continue

            if event_type not in EVENT_TYPES:
                event_type = "Pass"
            role = role_from_position(event.get("position", {}).get("name", ""))
            if role not in PLAYER_ROLES:
                role = "central_midfielder"
            lx, ly = bin_location(event.get("location"), self.cfg)
            goal = int(event_type == "Shot" and (event.get("shot") or {}).get("outcome", {}).get("name") in {"Goal", "Own Goal For"})
            if goal:
                if team == "home":
                    score_home += 1
                else:
                    score_away += 1
            old_possession = possession_team
            if event_type in POSSESSION_EVENT_TYPES:
                possession_team = team
            pflip = int(old_possession != possession_team)
            score_diff = clamp(score_home - score_away, self.cfg.score_diff_min, self.cfg.score_diff_max)

            ids.extend([
                self.vocab.id(f"EVT::{event_type}"),
                self.vocab.id(f"TEAM::{team}"),
                self.vocab.id(f"ROLE::{role}"),
                self.vocab.id(f"LX::{lx}"),
                self.vocab.id(f"LY::{ly}"),
                self.vocab.id(f"DT::{dt}"),
                self.vocab.id(f"SCORE::{score_diff}"),
                self.vocab.id(f"PFLIP::{pflip}"),
                self.vocab.id(f"GOAL::{goal}"),
                self.sep_id,
            ])
            frames.append({
                "minute": minute,
                "event_type": event_type,
                "team": team,
                "role": role,
                "lx": lx,
                "ly": ly,
                "dt": dt,
                "score_diff": score_diff,
                "possession_flip": pflip,
                "goal": goal,
            })
        ids.append(self.eos_id)
        return ids, frames

    def lineup_path_for_event_file(self, event_file: str, lineups_dir: Optional[str]) -> Optional[str]:
        if not lineups_dir:
            return None
        return str(Path(lineups_dir) / Path(event_file).name)
