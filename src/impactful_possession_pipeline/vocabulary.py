from typing import Dict, Iterable, List

from .config import TokenConfig, action_tokens_for_config, formation_state_tokens


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]

EVENT_TYPES = [
    "Pass",
    "Carry",
    "Dribble",
    "Ball Receipt*",
    "Duel",
    "Pressure",
    "Foul Committed",
    "Foul Won",
    "Shot",
    "Goal Keeper",
    "Interception",
    "Clearance",
    "Offside",
    "Error",
    "Half Start",
    "Half End",
    "Injury Stoppage",
    "Bad Behaviour",
    "Own Goal For",
    "Own Goal Against",
    "Penalty Conceded",
    "Miscontrol",
    "Ball Recovery",
    "Block",
    "Dispossessed",
]

PLAYER_ROLES = [
    "goalkeeper",
    "defender",
    "wingback",
    "defensive_midfielder",
    "central_midfielder",
    "attacking_midfielder",
    "winger",
    "forward",
]

FRAME_FIELDS = ["EVT", "TEAM", "ROLE", "LX", "LY", "DT", "SCORE", "PFLIP", "GOAL"]


class Vocab:
    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        for token in SPECIAL_TOKENS:
            self.add(token)

    def add(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def id(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])

    def token(self, idx: int) -> str:
        if 0 <= int(idx) < len(self.itos):
            return self.itos[int(idx)]
        return "<unk>"

    def __len__(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, obj: dict) -> "Vocab":
        vocab = cls()
        vocab.stoi.clear()
        vocab.itos.clear()
        for token in obj["itos"]:
            vocab.add(token)
        return vocab


def build_vocab(cfg: TokenConfig = TokenConfig(), extra_tokens: Iterable[str] = ()) -> Vocab:
    vocab = Vocab()
    for event_type in EVENT_TYPES:
        vocab.add(f"EVT::{event_type}")
    for team in ("home", "away"):
        vocab.add(f"TEAM::{team}")
    for role in PLAYER_ROLES:
        vocab.add(f"ROLE::{role}")
    for x in range(cfg.x_bins):
        vocab.add(f"LX::{x}")
    for y in range(cfg.y_bins):
        vocab.add(f"LY::{y}")
    for dt in range(cfg.max_dt_minutes + 1):
        vocab.add(f"DT::{dt}")
    for score in range(cfg.score_diff_min, cfg.score_diff_max + 1):
        vocab.add(f"SCORE::{score}")
    for flag in (0, 1):
        vocab.add(f"PFLIP::{flag}")
        vocab.add(f"GOAL::{flag}")
    for token in action_tokens_for_config(cfg):
        vocab.add(token)
    if cfg.emit_formation_state:
        for token in formation_state_tokens():
            vocab.add(token)
    for token in extra_tokens:
        vocab.add(token)
    return vocab
