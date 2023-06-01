from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import reduce
from typing import Any

from dex import itemdex, movedex, pokedex, typedex
from states.move_state import MoveState


@dataclass
class PokemonState:
    name: str
    identifier: str
    gen: int
    owner: str
    level: int
    gender: str | None
    hp: int
    max_hp: int
    status: str | None
    active: bool
    stats: dict[str, int]
    moves: list[MoveState]
    alt_moves: list[MoveState]
    ability: str | list[str]
    item: str | None
    from_opponent: bool
    alt_ability: str | None = None
    item_off: bool = False
    preparing: bool = False
    transformed: bool = False
    illusion: bool = False
    can_mega: bool = False
    can_zmove: bool = False
    can_max: bool = False
    maxed: bool = False

    ###################################################################################################################
    # Constructors

    @classmethod
    def from_request(cls, pokemon_json: Any, gen: int, owner: str) -> PokemonState:
        name = pokemon_json["ident"][4:]
        identifier = PokemonState.__get_identifier(name)
        level, gender = PokemonState.__parse_details(pokemon_json["details"])
        split_condition = pokemon_json["condition"].split()
        split_hp_frac = split_condition[0].split("/")
        hp = int(split_hp_frac[0])
        max_hp = int(split_hp_frac[1]) if pokemon_json["condition"] != "0 fnt" else 100
        status = split_condition[1] if len(split_condition) == 2 else None
        types = [t.lower() for t in pokedex[f"gen{gen}"][identifier]["types"]]
        moves = [MoveState.from_name(name, gen, "ghost" in types) for name in pokemon_json["moves"]]
        item = pokemon_json["item"] if pokemon_json["item"] != "" else None
        can_mega = (
            item is not None
            and "megaEvolves" in itemdex[f"gen{gen}"][item]
            and itemdex[f"gen{gen}"][item]["megaEvolves"] == name
        ) or (name == "Rayquaza" and "dragonascent" in pokemon_json["moves"] and gen in [6, 7])
        return cls(
            name=name,
            identifier=identifier,
            gen=gen,
            owner=owner,
            level=level,
            gender=gender,
            hp=hp,
            max_hp=max_hp,
            status=status,
            active=pokemon_json["active"],
            stats=pokemon_json["stats"],
            moves=moves,
            alt_moves=[],
            ability=pokemon_json["baseAbility"],
            item=item,
            from_opponent=False,
            can_mega=can_mega,
            can_zmove=item is not None and "zMove" in itemdex[f"gen{gen}"][item],
            can_max=gen == 8 and name not in ["Eternatus", "Zacian", "Zamazenta"],
        )

    @classmethod
    def from_protocol(cls, name: str, details: str, gen: int, owner: str) -> PokemonState:
        identifier = PokemonState.__get_identifier(name)
        level, gender = PokemonState.__parse_details(details)
        stats = pokedex[f"gen{gen}"][identifier]["baseStats"]
        abilities = pokedex[f"gen{gen}"][identifier]["abilities"].values()
        ability_identifiers = [PokemonState.__get_ability_identifier(ability) for ability in abilities]
        return cls(
            name=name,
            identifier=identifier,
            gen=gen,
            owner=owner,
            level=level,
            gender=gender,
            hp=100,
            max_hp=100,
            status=None,
            active=False,
            stats=stats,
            moves=[],
            alt_moves=[],
            ability=ability_identifiers[0] if len(ability_identifiers) == 1 else ability_identifiers,
            item=None,
            from_opponent=True,
        )

    ###################################################################################################################
    # Parsing/String Manipulation

    def __get_full_move_name(self, part_name: str):
        return (
            part_name
            if part_name != "Hidden Power"
            else [move.name for move in self.get_moves() if move.name[:12] == "Hidden Power"][0]
        )

    @staticmethod
    def __get_identifier(name: str) -> str:
        return re.sub(r"[\s\-\.\:\’]+", "", name).lower()

    @staticmethod
    def __parse_details(details: str) -> tuple[int, str | None]:
        # split_details format: "<alias>, <maybe level>, <maybe gender>, <maybe shiny>"
        # examples: "Castform, M, shiny", "Moltres, L84", "Raichu, L88, M"
        split_details = details.split(", ")
        if split_details[-1] == "shiny":
            split_details = split_details[:-1]
        if len(split_details) == 1:
            level = 100
            gender = None
        elif len(split_details) == 2:
            if split_details[1][0] == "L":
                level = int(split_details[1][1:])
                gender = None
            else:
                level = 100
                gender = split_details[1]
        else:
            level = int(split_details[1][1:])
            gender = split_details[2]
        return level, gender

    @staticmethod
    def parse_condition(condition: str) -> tuple[int, str | None]:
        split_condition = condition.split()
        split_hp_frac = split_condition[0].split("/")
        hp = int(split_hp_frac[0])
        status = split_condition[1] if len(split_condition) == 2 else None
        return hp, status

    @staticmethod
    def __get_ability_identifier(name: str) -> str:
        return re.sub(r"[\s\-]+", "", name).lower()

    @staticmethod
    def __get_item_identifier(name: str) -> str:
        return re.sub(r"[\s\-]+", "", name).lower()

    def get_json_str(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "id": self.identifier,
                "condition": f"{self.hp}/{self.max_hp}" + (f" {self.status}" if self.status else ""),
                "active": self.active,
                "stats": self.stats,
                "moves": [json.loads(move.get_json_str()) for move in self.get_moves()],
                "ability": self.__get_ability(),
                "item": self.item,
            }
        )

    ###################################################################################################################
    # Getter methods

    def get_moves(self) -> list[MoveState]:
        if self.transformed:
            return self.alt_moves
        elif "Mimic" in [move.name for move in self.moves] and len(self.alt_moves) == 1:
            return [move if move.name != "Mimic" else self.alt_moves[0] for move in self.moves]
        else:
            return self.moves

    def __get_last_used(self) -> MoveState | None:
        moves = [move for move in self.get_moves() if move.just_used]
        if len(moves) == 0:
            return None
        if len(moves) == 1:
            return moves[0]
        else:
            raise RuntimeError(f"Pokemon {self.name} cannot have move just used = {[move.name for move in moves]}")

    def __get_ability(self) -> str | list[str]:
        return self.alt_ability or self.ability

    def get_item(self) -> str | None:
        return None if self.item_off else self.item

    ###################################################################################################################
    # Setter methods

    def __update_last_used(self, name: str):
        last_last_moves = [move for move in (self.moves + self.alt_moves) if move.just_used]
        if len(last_last_moves) == 1:
            last_last_moves[0].just_used = False
        move_last_used = [move for move in self.get_moves() if move.name == name][0]
        move_last_used.just_used = True

    def update_special_options(self, mega_used: bool, zmove_used: bool, max_used: bool):
        if mega_used:
            self.can_mega = False
        if zmove_used:
            self.can_zmove = False
        if max_used:
            self.can_max = False

    ###################################################################################################################
    # Processes PokemonState object into a feature vector to be fed into the model's input layer

    def process(self) -> list[float]:
        gender_features = [
            float(gender_bool) for gender_bool in [self.gender == "M", self.gender == "F", self.gender == None]
        ]
        all_types = typedex[f"gen{self.gen}"].keys()
        types = [t.lower() for t in pokedex[f"gen{self.gen}"][self.identifier]["types"]]
        type_features = [float(t in types) for t in all_types]
        hp_features = [self.hp / self.max_hp] if self.from_opponent else [self.hp / self.max_hp, self.max_hp / 1000]
        status_conditions = ["psn", "tox", "par", "slp", "brn", "frz", "fnt"]
        status_features = [float(self.status == status_condition) for status_condition in status_conditions]
        stats = [stat / 255 if self.from_opponent else stat / 1000 for stat in self.stats.values()]
        move_feature_lists = [move.process() for move in self.get_moves()]
        move_feature_lists.extend([[0.0] * 22] * (4 - len(move_feature_lists)))
        move_features = reduce(lambda features1, features2: features1 + features2, move_feature_lists)
        features = gender_features + hp_features + status_features + stats + type_features + move_features
        return features

    ###################################################################################################################
    # Self-updating methods used when reading through the lines of the protocol

    def switch_in(self, hp: int, status: str | None):
        if self.status == "fnt":
            raise RuntimeError(f"Cannot switch in fainted pokemon {self.name}.")
        else:
            self.hp = hp
            self.status = status
            self.active = True
            for move in self.get_moves():
                move.update_item_disabled(None, self.get_item())

    def switch_out(self):
        if self.__get_ability() == "regenerator" and self.status != "fnt":
            self.hp = min(int(self.hp + self.max_hp / 3), self.max_hp)
        self.active = False
        for move in self.moves:
            move.disable_disabled = False
            move.encore_disabled = False
            move.taunt_disabled = False
            move.update_item_disabled(self.get_item(), None)
            move.self_disabled = False
        self.alt_moves = []
        self.alt_ability = None
        self.preparing = False
        self.transformed = False
        self.trapped = False

    def use_move(self, name: str, info: list[str], pressure: bool):
        full_name = self.__get_full_move_name(name)
        # Known "[from]" instances:
        #     "[from]lockedmove": pp isn't used when locked into a move
        #     "[from]move: <unowned_move>": this can happen when using a move like Copycat
        #     "[from]move: Sleep Talk": indicates that another move is being used due to Sleep Talk
        if (info and info[0][:6] == "[from]" and full_name not in info[0][6:]) or full_name == "Struggle":
            pass
        elif full_name in [move.name for move in self.get_moves()] + [f"Z-{move.name}" for move in self.get_moves()]:
            move = [move for move in self.get_moves() if full_name in [move.name, f"Z-{move.name}"]][0]
            pp_cost = self.__get_pp_cost(move, pressure)
            move.pp = max(0, move.pp - pp_cost)
            self.__update_last_used(move.name)
            for self_move in self.get_moves():
                self_move.update_item_disabled(self.get_item(), self.get_item())
                if self_move.name == "Gigaton Hammer":
                    self_move.self_disabled = move.name == "Gigaton Hammer"
        elif self.from_opponent:
            types = [t.lower() for t in pokedex[f"gen{self.gen}"][self.identifier]["types"]]
            new_move = MoveState.from_name(full_name, self.gen, "ghost" in types)
            pp_cost = self.__get_pp_cost(new_move, pressure)
            new_move.pp = max(0, new_move.pp - pp_cost)
            if self.transformed:
                self.alt_moves.append(new_move)
            else:
                self.moves.append(new_move)
            self.__update_last_used(new_move.name)
            for self_move in self.get_moves():
                self_move.update_item_disabled(self.get_item(), self.get_item())
                if self_move.name == "Gigaton Hammer":
                    self_move.self_disabled = new_move.name == "Gigaton Hammer"
        else:
            if "isZ" not in movedex[f"gen{self.gen}"][MoveState.get_identifier(full_name)] and full_name.split()[
                0
            ] not in ["Max", "G-Max"]:
                raise RuntimeError(
                    f"Chosen move {full_name} is not in pokemon {self.name}'s moveset: {[move.name for move in self.get_moves()]}."
                )

    def update_condition(self, hp: int, status: str | None):
        self.hp = hp
        self.status = status

    def update_ability(self, new_ability: str):
        self.alt_ability = PokemonState.__get_ability_identifier(new_ability)

    def update_item(self, new_item: str, info: list[str]):
        if not (info and info[0] == "[from] ability: Frisk"):
            new_item_identifier = PokemonState.__get_item_identifier(new_item)
            for move in self.get_moves():
                move.update_item_disabled(self.get_item(), None)
                move.update_item_disabled(None, new_item_identifier)
            self.item = new_item_identifier
            self.item_off = False

    def end_item(self, item: str, info: list[str]):
        item_identifier = PokemonState.__get_item_identifier(item)
        if self.get_item() == item_identifier:
            for move in self.get_moves():
                move.update_item_disabled(self.get_item(), None)
            if info and info[0] == "[from] move: Knock Off" and self.gen in [3, 4]:
                self.item_off = True
            else:
                self.item = None

    def start(self, info: list[str]):
        match info[0]:
            case "Disable":
                move = [move for move in self.get_moves() if move.name == self.__get_full_move_name(info[1])][0]
                move.disable_disabled = True
            case "Encore":
                last_used_move = self.__get_last_used()
                for move in self.get_moves():
                    if move != last_used_move:
                        move.encore_disabled = True
            case "Mimic":
                mimic_move = [move for move in self.get_moves() if move.name == "Mimic"][0]
                types = [t.lower() for t in pokedex[f"gen{self.gen}"][self.identifier]["types"]]
                new_move = MoveState.from_name(info[1], self.gen, "ghost" in types, from_mimic=True)
                if self.transformed:
                    mimic_move = new_move
                else:
                    mimic_move.self_disabled = True
                    self.alt_moves.append(new_move)
            case "move: Taunt":
                for move in self.get_moves():
                    if movedex[f"gen{self.gen}"][move.identifier]["category"] == "Status":
                        move.taunt_disabled = True
            case "item: Leppa Berry":
                move = [move for move in self.get_moves() if move.name == info[1]][0]
                move.pp = min(move.maxpp, 10)
            case "Dynamax":
                self.maxed = True
            case _:
                pass

    def end(self, info: list[str]):
        if info[0] == "Disable":
            move = [move for move in self.get_moves() if move.disable_disabled][0]
            move.disable_disabled = False
        elif info[0] == "Encore":
            for move in self.get_moves():
                move.encore_disabled = False
        elif info[0] == "move: Taunt":
            for move in self.get_moves():
                move.taunt_disabled = False
        elif info[0] == "Dynamax":
            self.maxed = False

    ###################################################################################################################
    # Helper methods

    def __get_pp_cost(self, move: MoveState, pressure: bool) -> int:
        if pressure:
            if move.get_category() != "Status" or move.target in ["all", "normal"]:
                pp_used = 2
            elif move.name in ["Imprison", "Snatch", "Spikes", "Stealth Rock", "Toxic Spikes"]:
                if self.gen <= 4:
                    pp_used = 1
                else:
                    pp_used = 2
            else:
                pp_used = 1
        else:
            pp_used = 1
        return pp_used
