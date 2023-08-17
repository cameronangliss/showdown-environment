import json
import re
from typing import Any

from poke_dojo.data.dex import movedex, typedex


class Move:
    name: str
    identifier: str
    __gen: int
    pp: int
    maxpp: int
    __target: str
    just_used: bool = False
    disable_disabled: bool = False
    bide_disabled: bool = False
    encore_disabled: bool = False
    taunt_disabled: bool = False
    __item_disabled: bool = False
    self_disabled: bool = False

    def __init__(self, name: str, gen: int, is_ghost: bool, from_mimic: bool = False):
        self.identifier = Move.get_identifier(name)
        details = movedex[f"gen{gen}"][self.identifier]
        self.name = details["name"]
        self.__gen = gen
        if from_mimic and gen > 2:
            pp = details["pp"]
        elif gen <= 2:
            pp = min(int(1.6 * details["pp"]), 61) if details["pp"] > 1 else 1
        else:
            pp = int(1.6 * details["pp"]) if details["pp"] > 1 else 1
        self.pp = pp
        self.maxpp = pp
        self.__target = (
            details["nonGhostTarget"]
            if "nonGhostTarget" in details and details["nonGhostTarget"] and not is_ghost
            else details["target"]
        )

    ###################################################################################################################
    # Getter methods

    @staticmethod
    def get_identifier(name: str) -> str:
        return re.sub(r"([\s\-\']+)|(\d+$)", "", name).lower()

    def is_disabled(self, maxed: bool = False) -> bool:
        return (
            self.pp == 0
            or (
                (
                    self.disable_disabled
                    or self.bide_disabled
                    or self.encore_disabled
                    or self.taunt_disabled
                    or self.self_disabled
                )
                and not maxed
            )
            or self.__item_disabled
        )

    def __get_category(self) -> str:
        return movedex[f"gen{self.__gen}"][self.identifier]["category"]

    def get_json_str(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "id": self.identifier,
                "pp": self.pp,
                "maxpp": self.maxpp,
                "target": self.__target,
                "disabled": self.is_disabled(),
            }
        )

    def get_pp_used(self, pressure: bool) -> int:
        if pressure:
            if self.__target in [
                "all",
                "allAdjacent",
                "allAdjacentFoes",
                "any",
                "normal",
                "randomNormal",
                "scripted",
            ]:
                pp_used = 2
            elif self.name in ["Imprison", "Snatch", "Spikes", "Stealth Rock", "Toxic Spikes"]:
                pp_used = 2
            else:
                pp_used = 1
        else:
            pp_used = 1
        return pp_used

    ###################################################################################################################
    # Setter methods

    def add_item(self, item: str, tricking: bool = False, maxed: bool = False):
        if item in ["choiceband", "choicescarf", "choicespecs"] and tricking:
            self.__item_disabled = not (maxed or self.just_used)
        elif item == "assaultvest" and self.__get_category() == "Status":
            self.__item_disabled = True
        elif self.__item_disabled and item[-5:] == "berry":
            self.__item_disabled = False

    def remove_item(self):
        self.__item_disabled = self.name == "Stuff Cheeks"

    def keep_item(self, item: str | None, maxed: bool = False):
        if item in ["choiceband", "choicescarf", "choicespecs"]:
            self.__item_disabled = not (maxed or self.just_used)

    def update_item(self, old_item: str | None, new_item: str, tricking: bool = False, maxed: bool = False):
        choice_items = ["choiceband", "choicescarf", "choicespecs"]
        if not (old_item in choice_items and new_item in choice_items):
            if old_item is not None:
                self.remove_item()
            self.add_item(new_item, tricking=tricking, maxed=maxed)

    ###################################################################################################################
    # Consistency checking

    def check_consistency(self, move_info: Any, zmove_pp_needs_update: Any, maxed: bool, just_unmaxed: bool):
        if "move" in move_info:
            identifier = self.get_identifier(move_info["move"])
            assert self.identifier == identifier, f"{self.identifier} != {identifier}"
        if "pp" in move_info:
            if zmove_pp_needs_update or maxed or just_unmaxed or self.__gen <= 3:
                self.pp = move_info["pp"]
            else:
                assert self.pp == move_info["pp"], f"{self.identifier}: {self.pp} != {move_info['pp']}"
        if "maxpp" in move_info:
            assert self.maxpp == move_info["maxpp"], f"{self.identifier}: {self.maxpp} != {move_info['maxpp']}"
        if "target" in move_info:
            assert self.__target == move_info["target"], f"{self.identifier}: {self.__target} != {move_info['target']}"
        if "disabled" in move_info:
            assert (
                self.is_disabled() == move_info["disabled"]
            ), f"{self.identifier}: {self.is_disabled()} != {move_info['disabled']}"

    ###################################################################################################################
    # Processes MoveState object into a feature vector to be fed into the model's input layer

    def process(self) -> list[float]:
        pp_frac_feature = self.pp / self.maxpp
        disabled_feature = float(self.is_disabled())
        details = movedex[f"gen{self.__gen}"][self.identifier]
        power_feature = details["basePower"] / 250
        accuracy_feature = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
        all_types = typedex[f"gen{self.__gen}"].keys()
        move_type = details["type"].lower()
        type_features = [float(t == move_type) for t in all_types]
        return [pp_frac_feature, disabled_feature, power_feature, accuracy_feature] + type_features
