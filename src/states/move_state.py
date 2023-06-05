import json
import re

from dex import movedex, typedex


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

    def __init__(self, name: str, gen: int, is_ghost: bool, copied: bool = False):
        self.identifier = MoveState.get_identifier(name)
        details = movedex[f"gen{gen}"][self.identifier]
        self.name = details["name"]
        self.gen = gen
        if copied:
            pp = details["pp"]
        elif gen == 1 or gen == 2:
            pp = min(int(1.6 * details["pp"]), 61) if details["pp"] > 1 else 1
        else:
            pp = int(1.6 * details["pp"]) if details["pp"] > 1 else 1
        self.pp = pp
        self.maxpp = pp
        self.target = (
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
            or (self.disable_disabled and not maxed)
            or (self.encore_disabled and not maxed)
            or (self.taunt_disabled and not maxed)
            or self.item_disabled
            or (self.self_disabled and not maxed)
        )

    def __get_category(self) -> str:
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

    def get_pp_used(self, pressure: bool) -> int:
        if not pressure:
            pp_used = 1
        else:
            if self.target in ["all", "allAdjacent", "allAdjacentFoes", "any", "normal", "randomNormal", "scripted"]:
                pp_used = 2
            elif self.name in ["Imprison", "Snatch", "Spikes", "Stealth Rock", "Toxic Spikes"]:
                if self.gen <= 4:
                    pp_used = 1
                else:
                    pp_used = 2
            else:
                pp_used = 1
        return pp_used

    def add_item(self, item: str):
        if item == "assaultvest" and self.__get_category() == "Status":
            self.item_disabled = True
        elif self.item_disabled and item[-5:] == "berry":
            self.item_disabled = False

    def remove_item(self):
        self.item_disabled = self.name == "Stuff Cheeks"

    def keep_item(self, item: str | None, maxed: bool = False):
        if item in ["choiceband", "choicescarf", "choicespecs"]:
            self.item_disabled = not (maxed or self.just_used)

    def swap_item(self, old_item: str, new_item: str, tricking: bool = False, maxed: bool = False):
        choice_items = ["choiceband", "choicescarf", "choicespecs"]
        if not tricking:
            if not (old_item in choice_items and new_item in choice_items):
                self.remove_item()
                self.add_item(new_item)
        else:
            if new_item not in choice_items:
                self.remove_item()
                self.add_item(new_item)
            elif old_item not in choice_items:
                self.remove_item()
                self.item_disabled = not (maxed or self.just_used)

    ###################################################################################################################
    # Processes MoveState object into a feature vector to be fed into the model's input layer

    def process(self) -> list[float]:
        pp_frac_feature = self.pp / self.maxpp
        disabled_feature = float(self.is_disabled())
        details = movedex[f"gen{self.gen}"][self.identifier]
        power_feature = details["basePower"] / 250
        accuracy_feature = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
        all_types = typedex[f"gen{self.gen}"].keys()
        move_type = details["type"].lower()
        type_features = [float(t == move_type) for t in all_types]
        return [pp_frac_feature, disabled_feature, power_feature, accuracy_feature] + type_features
