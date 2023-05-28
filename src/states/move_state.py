from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dex import itemdex, movedex, typedex


@dataclass
class MoveState:
    name: str
    identifier: str
    gen: int
    move_type: str
    pp: int
    maxpp: int
    category: str
    target: str
    just_used: bool
    disable_disabled: bool
    encore_disabled: bool
    taunt_disabled: bool
    item_disabled: bool
    no_item_disabled: bool
    self_disabled: bool

    @staticmethod
    def get_identifier(name: str) -> str:
        return re.sub(r"([\s\-\']+)|(\d+$)", "", name).lower()

    def is_disabled(self) -> bool:
        return (
            self.disable_disabled
            or self.encore_disabled
            or self.taunt_disabled
            or self.item_disabled
            or self.no_item_disabled
            or self.self_disabled
            or self.pp == 0
        )

    @classmethod
    def from_request(cls, move_json: Any, gen: int) -> MoveState:
        details = movedex[f"gen{gen}"][move_json["id"]]
        return cls(
            name=move_json["move"],
            identifier=MoveState.get_identifier(move_json["move"]),
            gen=gen,
            move_type=details["type"].lower(),
            pp=move_json["pp"],
            maxpp=move_json["maxpp"],
            category=details["category"],
            target=move_json["target"],
            just_used=False,
            disable_disabled=False,
            encore_disabled=False,
            taunt_disabled=False,
            item_disabled=False,
            no_item_disabled=False,
            self_disabled=False,
        )

    @classmethod
    def from_name(cls, name: str, gen: int, is_ghost: bool, from_mimic: bool = False) -> MoveState:
        identifier = MoveState.get_identifier(name)
        details = movedex[f"gen{gen}"][identifier]
        if from_mimic:
            pp = details["pp"]
        elif gen == 1 or gen == 2:
            pp = min(int(1.6 * details["pp"]), 61) if details["pp"] > 1 else 1
        else:
            pp = int(1.6 * details["pp"]) if details["pp"] > 1 else 1
        target = (
            details["nonGhostTarget"]
            if "nonGhostTarget" in details and details["nonGhostTarget"] and not is_ghost
            else details["target"]
        )
        return cls(
            name=details["name"],
            identifier=identifier,
            gen=gen,
            move_type=details["type"].lower(),
            pp=pp,
            maxpp=pp,
            category=details["category"],
            target=target,
            just_used=False,
            disable_disabled=False,
            encore_disabled=False,
            taunt_disabled=False,
            item_disabled=False,
            no_item_disabled=False,
            self_disabled=False,
        )

    def can_zmove(self, item: str | None) -> bool:
        return (
            item is not None
            and "zMoveType" in itemdex[f"gen{self.gen}"][item]
            and itemdex[f"gen{self.gen}"][item]["zMoveType"] == movedex[f"gen{self.gen}"][self.identifier]["type"]
        )

    def process(self) -> list[float]:
        pp_frac_feature = self.pp / self.maxpp
        disabled_feature = float(self.is_disabled())
        details = movedex[f"gen{self.gen}"][self.identifier]
        power_feature = details["basePower"] / 250
        accuracy_feature = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
        types = typedex[f"gen{self.gen}"].keys()
        move_type = details["type"].lower()
        type_features = [float(t == move_type) for t in types]
        return [pp_frac_feature, disabled_feature, power_feature, accuracy_feature] + type_features
