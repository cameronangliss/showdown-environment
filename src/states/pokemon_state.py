from __future__ import annotations

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
    types: list[str]
    hp: int
    max_hp: int
    status: str | None
    active: bool
    stats: dict[str, int]
    moves: list[MoveState]
    alt_moves: list[MoveState]
    max_moves: list[MoveState]
    ability: str | list[str]
    alt_ability: str | None
    item: str | None
    item_off: bool
    preparing: bool
    transformed: bool
    illusion: bool
    from_enemy: bool
    can_mega_evo: bool
    can_zmove: bool
    can_max: bool
    maxed: bool

    @staticmethod
    def concat_features(list1: list[float], list2: list[float]) -> list[float]:
        return list1 + list2

    @staticmethod
    def get_identifier(name: str) -> str:
        return re.sub(r"[\s\-\.\:\’]+", "", name).lower()

    def get_full_move_name(self, part_name: str):
        return (
            part_name
            if (self.from_enemy or part_name != "Hidden Power")
            else [move.name for move in self.get_moves() if move.name[:12] == "Hidden Power"][0]
        )

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
        split_condition = condition.split()
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

    def get_item(self) -> str | None:
        return None if self.item_off else self.item

    @classmethod
    def from_request(cls, pokemon_json: Any, gen: int, owner: str) -> PokemonState:
        name = pokemon_json["ident"][4:]
        identifier = PokemonState.get_identifier(name)
        level, gender = PokemonState.parse_details(pokemon_json["details"])
        types = [t.lower() for t in pokedex[f"gen{gen}"][identifier]["types"]]
        split_condition = pokemon_json["condition"].split()
        split_hp_frac = split_condition[0].split("/")
        hp = int(split_hp_frac[0])
        max_hp = int(split_hp_frac[1]) if pokemon_json["condition"] != "0 fnt" else 100
        status = split_condition[1] if len(split_condition) == 2 else None
        moves = [MoveState.from_name(name, gen, "ghost" in types) for name in pokemon_json["moves"]]
        item = pokemon_json["item"] if pokemon_json["item"] != "" else None
        can_mega_evo = (
            item is not None
            and "megaEvolves" in itemdex[f"gen{gen}"][item]
            and itemdex[f"gen{gen}"][item]["megaEvolves"] == name
        )
        return cls(
            name=name,
            identifier=identifier,
            gen=gen,
            owner=owner,
            level=level,
            gender=gender,
            types=types,
            hp=hp,
            max_hp=max_hp,
            status=status,
            active=pokemon_json["active"],
            stats=pokemon_json["stats"],
            moves=moves,
            alt_moves=[],
            max_moves=[],
            ability=pokemon_json["baseAbility"],
            alt_ability=None,
            item=item,
            item_off=False,
            preparing=False,
            transformed=False,
            illusion=False,
            from_enemy=False,
            can_mega_evo=can_mega_evo,
            can_zmove=item is not None and "zMove" in itemdex[f"gen{gen}"][item],
            can_max=gen == 8,
            maxed=False,
        )

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
            max_moves=[],
            ability=ability_identifiers[0] if len(ability_identifiers) == 1 else ability_identifiers,
            alt_ability=None,
            item=None,
            item_off=False,
            preparing=False,
            transformed=False,
            illusion=False,
            from_enemy=True,
            can_mega_evo=False,
            can_zmove=False,
            can_max=False,
            maxed=False,
        )

    def update_special_options(self, mega_used: bool, zmove_used: bool, max_used: bool):
        if mega_used:
            self.can_mega_evo = False
        if zmove_used:
            self.can_zmove = False
        if max_used:
            self.can_max = False

    def update_last_used(self, name: str):
        last_last_moves = [move for move in (self.moves + self.alt_moves) if move.just_used]
        if len(last_last_moves) == 1:
            last_last_moves[0].just_used = False
        move_last_used = [move for move in self.get_moves() if move.name == name][0]
        move_last_used.just_used = True

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
            move.self_disabled = False
        self.alt_moves = []
        self.alt_ability = None
        self.preparing = False
        self.transformed = False
        self.trapped = False

    def use_move(self, name: str, info: list[str], pressure: bool):
        full_name = self.get_full_move_name(name)
        # Known "[from]" instances:
        #     "[from]lockedmove": pp isn't used when locked into a move
        #     "[from]move: <unowned_move>": this can happen when using a move like Copycat
        #     "[from]move: Sleep Talk": indicates that another move is being used due to Sleep Talk
        if (info and info[0][:6] == "[from]" and full_name not in info[0][6:]) or full_name == "Struggle":
            pass
        elif full_name in [move.name for move in self.get_moves()]:
            move = [move for move in self.get_moves() if move.name == full_name][0]
            pp_cost = self.get_pp_cost(move, pressure)
            if move.pp > 0:
                move.pp = max(0, move.pp - pp_cost)
            else:
                raise RuntimeError(f"Move {move.name} of pokemon {self.name} has no more power points.")
            self.update_last_used(full_name)
            self.disable_moves_with_item()
            for self_move in self.get_moves():
                if self_move.name == "Gigaton Hammer":
                    self_move.self_disabled = move.name == "Gigaton Hammer"
        elif self.from_enemy:
            new_move = MoveState.from_name(full_name, self.gen, "ghost" in self.types)
            pp_cost = self.get_pp_cost(new_move, pressure)
            new_move.pp = max(0, new_move.pp - pp_cost)
            if self.transformed:
                self.alt_moves.append(new_move)
            else:
                self.moves.append(new_move)
            self.update_last_used(full_name)
            self.disable_moves_with_item()
            for self_move in self.get_moves():
                if self_move.name == "Gigaton Hammer":
                    self_move.self_disabled = new_move.name == "Gigaton Hammer"
        else:
            if (
                full_name.split()[0] not in ["Max", "G-Max"]
                and "isZ" not in movedex[f"gen{self.gen}"][MoveState.get_identifier(full_name)]
                and full_name != "Z-Roost"
            ):
                raise RuntimeError(
                    f"Chosen move {full_name} is not in pokemon {self.name}'s moveset: {[move.name for move in self.get_moves()]}."
                )

    def get_pp_cost(self, move: MoveState, pressure: bool) -> int:
        if pressure:
            if move.category != "Status" or move.target in ["all", "normal"]:
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

    def disable_moves_with_item(self):
        if self.get_item() in ["choiceband", "choicescarf", "choicespecs"]:
            last_used = self.get_last_used()
            if last_used:
                other_moves = [move for move in self.get_moves() if move.name != last_used.name]
                for move in other_moves:
                    move.item_disabled = True

    def update_condition(self, hp: int, status: str | None):
        self.hp = hp
        self.status = status

    def update_ability(self, new_ability: str):
        self.alt_ability = PokemonState.get_ability_identifier(new_ability)

    def update_item(self, new_item: str, info: list[str]):
        new_item_identifier = PokemonState.get_item_identifier(new_item)
        if info and info[0] == "[from] ability: Frisk":
            pass
        else:
            self.update_moves_item_disabled(new_item_identifier)
            self.item_off = False
        self.update_moves_no_item_disabled()

    def end_item(self, item: str, info: list[str]):
        item_identifier = PokemonState.get_item_identifier(item)
        if self.get_item() == item_identifier:
            if info and info[0] == "[from] move: Knock Off":
                temp = self.get_item()
                self.update_moves_item_disabled(None)
                if self.gen in [3, 4]:
                    self.item = temp
                    self.item_off = True
            else:
                self.update_moves_item_disabled(None)
        self.update_moves_no_item_disabled()

    def update_moves_item_disabled(self, new_item: str | None):
        if self.get_item() in ["assaultvest", "choiceband", "choicescarf", "choicespecs"]:
            for move in self.get_moves():
                move.item_disabled = False
        if new_item == "assaultvest":
            for move in self.get_moves():
                if move.category == "Status":
                    move.item_disabled = True
        self.item = new_item

    def update_moves_no_item_disabled(self):
        for move in self.get_moves():
            if move.name == "Stuff Cheeks":
                current_item = self.get_item()
                move.no_item_disabled = current_item is None or current_item[-5:] != "berry"

    def transform(self):
        self.transformed = True

    def start(self, info: list[str]):
        match info[0]:
            case "Disable":
                move = [move for move in self.get_moves() if move.name == self.get_full_move_name(info[1])][0]
                move.disable_disabled = True
            case "Encore":
                last_used_move = self.get_last_used()
                for move in self.get_moves():
                    if move != last_used_move:
                        move.encore_disabled = True
            case "move: Mimic":
                mimic_move = [move for move in self.get_moves() if move.name == "Mimic"][0]
                new_move = MoveState.from_name(info[1], self.gen, "ghost" in self.types, from_mimic=True)
                if self.transformed:
                    mimic_move = new_move
                else:
                    mimic_move.self_disabled = True
                    self.alt_moves.append(new_move)
            case "move: Taunt":
                for move in self.get_moves():
                    if move.category == "Status":
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
