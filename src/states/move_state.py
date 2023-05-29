from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dex import movedex, typedex


@dataclass
class MoveState:
    name: str
    identifier: str
    gen: int
    pp: int
    maxpp: int
    target: str
    just_used: bool = False
    disable_disabled: bool = False
    encore_disabled: bool = False
    taunt_disabled: bool = False
    item_disabled: bool = False
    no_item_disabled: bool = False
    self_disabled: bool = False

    ###################################################################################################################
    # Constructors

    @classmethod
    def from_request(cls, move_json: Any, gen: int, pokemon_item: str | None) -> MoveState:
        return cls(
            name=move_json["move"],
            identifier=MoveState.get_identifier(move_json["move"]),
            gen=gen,
            pp=move_json["pp"],
            maxpp=move_json["maxpp"],
            target=move_json["target"],
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
            pp=pp,
            maxpp=pp,
            target=target,
        )

    ###################################################################################################################
    # Getter methods

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

    ###################################################################################################################
    # Processes MoveState object into a feature vector to be fed into the model's input layer

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
