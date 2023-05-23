from __future__ import annotations

import re
from dataclasses import dataclass
from functools import reduce
from typing import Any

from dex import pokedex, typedex
from states.move_state import MoveState


@dataclass
class PokemonState:
    name: str
    identifier: str
    gen: int
    owner: str
    level: int
    gender: str | None
    types: list[str]
    hp: int
    max_hp: int
    status: str | None
    active: bool
    stats: dict[str, int]
    moves: list[MoveState]
    alt_moves: list[MoveState]
    ability: str | list[str]
    alt_ability: str | None
    item: str | None
    transformed: bool
    illusion: bool
    from_enemy: bool

    @staticmethod
    def concat_features(list1: list[float], list2: list[float]) -> list[float]:
        return list1 + list2

    @staticmethod
    def get_identifier(name: str) -> str:
        return re.sub(r"[\s\-\.\:\’]+", "", name).lower()

    @staticmethod
    def get_ability_identifier(name: str) -> str:
        return re.sub(r"[\s\-]+", "", name).lower()

    @staticmethod
    def get_item_identifier(name: str) -> str:
        return re.sub(r"[\s\-]+", "", name).lower()

    @staticmethod
    def parse_details(details: str) -> tuple[int, str | None]:
        split_details = details.split(", ")
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
        split_condition = condition.split(" ")
        split_hp_frac = split_condition[0].split("/")
        hp = int(split_hp_frac[0])
        status = split_condition[1] if len(split_condition) == 2 else None
        return hp, status

    def get_moves(self) -> list[MoveState]:
        if self.transformed:
            return self.alt_moves
        else:
            return self.moves + self.alt_moves

    def get_last_used(self) -> MoveState | None:
        moves = [move for move in self.get_moves() if move.just_used]
        if len(moves) == 0:
            return None
        if len(moves) == 1:
            return moves[0]
        else:
            raise RuntimeError(f"Pokemon {self.name} cannot have move just used = {[move.name for move in moves]}")

    def get_ability(self) -> str | list[str]:
        return self.alt_ability or self.ability

    @classmethod
    def from_request(cls, pokemon_json: Any, gen: int, owner: str) -> PokemonState:
        self = cls("", "", gen, owner, 0, None, [], 0, 0, None, False, {}, [], [], "", None, None, False, False, False)
        self.name = pokemon_json["ident"][4:]
        self.identifier = PokemonState.get_identifier(self.name)
        level, gender = PokemonState.parse_details(pokemon_json["details"])
        self.level = level
        self.gender = gender
        self.types = [t.lower() for t in pokedex[f"gen{gen}"][self.identifier]["types"]]
        split_condition = pokemon_json["condition"].split(" ")
        split_hp_frac = split_condition[0].split("/")
        self.hp = int(split_hp_frac[0])
        self.max_hp = int(split_hp_frac[1]) if pokemon_json["condition"] != "0 fnt" else 100
        self.status = split_condition[1] if len(split_condition) == 2 else None
        self.active = pokemon_json["active"]
        self.stats = pokemon_json["stats"]
        self.moves = [MoveState.from_name(name, gen, "ghost" in self.types) for name in pokemon_json["moves"]]
        self.ability = pokemon_json["baseAbility"]
        self.item = pokemon_json["item"] if pokemon_json["item"] != "" else None
        return self

    @classmethod
    def from_protocol(cls, name: str, details: str, gen: int, owner: str) -> PokemonState:
        identifier = PokemonState.get_identifier(name)
        level, gender = PokemonState.parse_details(details)
        stats = pokedex[f"gen{gen}"][identifier]["baseStats"]
        abilities = pokedex[f"gen{gen}"][identifier]["abilities"].values()
        ability_identifiers = [PokemonState.get_ability_identifier(ability) for ability in abilities]
        return cls(
            name=name,
            identifier=identifier,
            gen=gen,
            owner=owner,
            level=level,
            gender=gender,
            types=[t.lower() for t in pokedex[f"gen{gen}"][identifier]["types"]],
            hp=100,
            max_hp=100,
            status=None,
            active=False,
            stats=stats,
            moves=[],
            alt_moves=[],
            ability=ability_identifiers[0] if len(ability_identifiers) == 1 else ability_identifiers,
            alt_ability=None,
            item=None,
            transformed=False,
            illusion=False,
            from_enemy=True,
        )

    def update_last_used(self, name: str):
        last_last_moves = [move for move in (self.moves + self.alt_moves) if move.just_used == True]
        if len(last_last_moves) == 1:
            last_last_moves[0].just_used = False
        move_last_used = [move for move in self.get_moves() if move.name == name][0]
        move_last_used.just_used = True

    def update_moves_disabled(self):
        for move in self.get_moves():
            move.disabled = move.is_disabled()

    def switch_in(self, hp: int, status: str | None):
        if self.status == "fnt":
            raise RuntimeError(f"Cannot switch in fainted pokemon {self.name}.")
        else:
            self.hp = hp
            self.status = status
            self.active = True

    def switch_out(self):
        if self.get_ability() == "regenerator" and self.status != "fnt":
            self.hp = min(int(self.hp + self.max_hp / 3), self.max_hp)
        self.active = False
        for move in self.moves:
            move.disable_disabled = False
            move.encore_disabled = False
            move.taunt_disabled = False
            move.item_disabled = False
            move.disabled = move.is_disabled()
        self.alt_moves = []
        self.alt_ability = None
        self.transformed = False
        self.trapped = False

    def use_move(self, name: str, message: str | None, pressure: bool):
        name = (
            name
            if (self.from_enemy or name != "Hidden Power")
            else [move.name for move in self.get_moves() if move.name[:12] == "Hidden Power"][0]
        )
        if (message is not None and "[from]" in message and name not in message) or name == "Struggle":
            used_owned_move = False
        elif name in [move.name for move in self.get_moves()]:
            move = [move for move in self.get_moves() if move.name == name and not move.disabled][0]
            pp_cost = self.get_pp_cost(move, message, pressure)
            if move.pp > 0:
                move.pp = max(0, move.pp - pp_cost)
            else:
                raise RuntimeError(f"Move {move.name} of pokemon {self.name} has no more power points.")
            used_owned_move = True
        elif self.from_enemy:
            new_move = MoveState.from_name(name, self.gen, "ghost" in self.types)
            pp_cost = self.get_pp_cost(new_move, message, pressure)
            new_move.pp = max(0, new_move.pp - pp_cost)
            if self.transformed:
                self.alt_moves.append(new_move)
            else:
                self.moves.append(new_move)
            used_owned_move = True
        else:
            raise RuntimeError(f"Chosen move {name} is not in pokemon {self.name}'s moveset: {self.get_moves()}.")
        if used_owned_move:
            self.update_last_used(name)
            if self.item in ["choiceband", "choicescarf", "choicespecs"]:
                last_used = self.get_last_used()
                if last_used:
                    other_moves = [move for move in self.get_moves() if move.name != last_used.name]
                    for move in other_moves:
                        move.item_disabled = True
        self.update_moves_disabled()

    def get_pp_cost(self, move: MoveState, message: str | None, pressure: bool) -> int:
        if message and message == "[from]lockedmove":
            pp_used = 0
        elif pressure:
            if move.category != "Status" or move.target in ["all", "normal"]:
                pp_used = 2
            elif self.gen <= 4:
                pp_used = 1
            elif move.name in ["Imprison", "Snatch", "Spikes", "Stealth Rock", "Toxic Spikes"]:
                pp_used = 2
            else:
                pp_used = 1
        else:
            pp_used = 1
        return pp_used

    def update_condition(self, hp: int, status: str | None):
        self.hp = hp
        self.status = status

    def update_ability(self, new_ability: str):
        self.alt_ability = PokemonState.get_ability_identifier(new_ability)

    def update_item(self, new_item: str | None):
        old_item = self.item
        if old_item in ["choiceband", "choicescarf", "choicespecs"]:
            for move in self.get_moves():
                move.item_disabled = False
        self.update_moves_disabled()
        if new_item:
            new_item_identifier = PokemonState.get_item_identifier(new_item)
            self.item = new_item_identifier
        else:
            self.item = None
        for move in self.get_moves():
            if move.name == "Stuff Cheeks" and (self.item is None or self.item[-5:] != "berry"):
                move.no_item_disabled = True
            elif move.name == "Stuff Cheeks" and self.item is not None and self.item[-5:] == "berry":
                move.no_item_disabled = False

    def transform(self):
        self.transformed = True

    def start(self, info: list[str]):
        if info[0] == "Disable":
            move = [move for move in self.get_moves() if move.name == info[1]][0]
            move.disable_disabled = True
        elif info[0] == "Encore":
            last_used_move = self.get_last_used()
            for move in self.get_moves():
                if move != last_used_move:
                    move.encore_disabled = True
        elif info[0][-5:] == "Mimic":
            mimic_move = [move for move in self.get_moves() if move.name == "Mimic"][0]
            new_move = MoveState.from_name(info[1], self.gen, "ghost" in self.types, from_mimic=True)
            if self.transformed:
                mimic_move = new_move
            else:
                mimic_move.disabled = True
                self.alt_moves.append(new_move)
        elif info[0] == "move: Taunt":
            for move in self.get_moves():
                if move.category == "Status":
                    move.taunt_disabled = True
        self.update_moves_disabled()

    def end(self, info: list[str]):
        if info[0] == "Disable":
            move = [move for move in self.get_moves() if move.disable_disabled == True][0]
            move.disable_disabled = False
        elif info[0] == "Encore":
            for move in self.get_moves():
                move.encore_disabled = False
        elif info[0] == "move: Taunt":
            for move in self.get_moves():
                move.taunt_disabled = False
        self.update_moves_disabled()

    def process(self) -> list[float]:
        gender_features = [
            float(gender_bool) for gender_bool in [self.gender == "M", self.gender == "F", self.gender == None]
        ]
        types = typedex[f"gen{self.gen}"].keys()
        type_features = [float(t in self.types) for t in types]
        hp_features = [self.hp / self.max_hp] if self.from_enemy else [self.hp / self.max_hp, self.max_hp / 1000]
        status_conditions = ["psn", "tox", "par", "slp", "brn", "frz", "fnt"]
        status_features = [float(self.status == status_condition) for status_condition in status_conditions]
        stats = [stat / 255 if self.from_enemy else stat / 1000 for stat in self.stats.values()]
        move_feature_lists = [move.process() for move in self.moves]
        move_feature_lists.extend([[0.0] * 22] * (8 - len(move_feature_lists)))
        move_features = reduce(PokemonState.concat_features, move_feature_lists)
        alt_move_feature_lists = [move.process() for move in self.alt_moves]
        alt_move_feature_lists.extend([[0.0] * 22] * (4 - len(alt_move_feature_lists)))
        alt_move_features = reduce(PokemonState.concat_features, alt_move_feature_lists)
        features = (
            gender_features + hp_features + status_features + stats + type_features + move_features + alt_move_features
        )
        return features
