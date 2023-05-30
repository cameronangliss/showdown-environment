from __future__ import annotations

import json
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
            self.pp == 0
            or self.disable_disabled
            or self.encore_disabled
            or self.taunt_disabled
            or self.item_disabled
            or self.self_disabled
        )

    def get_category(self) -> str:
        return movedex[f"gen{self.gen}"][self.identifier]["category"]

    def get_json_str(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "id": self.identifier,
                "pp": self.pp,
                "maxpp": self.maxpp,
                "target": self.target,
                "disabled": self.is_disabled(),
            }
        )

    ###################################################################################################################
    # Setter methods

    def update_item_disabled(self, old_item: str | None, new_item: str | None):
        # adding item
        if old_item is None and new_item is not None:
            if new_item == "assaultvest" and self.get_category() == "Status":
                self.item_disabled = True
            elif self.item_disabled and new_item[-5:] == "berry":
                self.item_disabled = False
        # removing item
        elif old_item is not None and new_item is None:
            self.item_disabled = self.name == "Stuff Cheeks"
        # keep item
        elif old_item == new_item:
            item = old_item
            if item in ["choiceband", "choicescarf", "choicespecs"]:
                self.item_disabled = not self.just_used
        else:
            raise RuntimeError(
                f"Must either add, remove, or keep current item. None are the case with old_item = {old_item} and "
                f"new_item = {new_item}."
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
