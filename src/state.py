from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import reduce
from typing import Any

import torch

from dex import movedex, pokedex, typedex


def concat_features(list1: list[float], list2: list[float]) -> list[float]:
    return list1 + list2


@dataclass
class MoveState:
    move: str
    move_id: str
    pp: int
    maxpp: int
    target: str
    disabled: bool

    @staticmethod
    def get_id(move: str) -> str:
        return re.sub(r"([\s\-\']+)|(\d+$)", "", move).lower()

    @classmethod
    def from_name(cls, move: str) -> MoveState:
        move_id = MoveState.get_id(move)
        details = movedex[move_id]
        return cls(
            move=details["name"],
            move_id=move_id,
            pp=int(1.6 * details["pp"]) if details["pp"] > 1 else 1,
            maxpp=int(1.6 * details["pp"]) if details["pp"] > 1 else 1,
            target=details["target"],
            disabled=False,
        )

    def update(self, active_moves: list[Any]):
        for active_move in active_moves:
            if active_move and self.move == active_move["move"] and "pp" in active_move:
                self.pp = active_move["pp"]
                self.disabled = active_move["disabled"]

    def process(self) -> list[float]:
        pp_frac = self.pp / self.maxpp
        disabled = float(self.disabled)
        details = movedex[self.move_id]
        power = details["basePower"] / 250
        accuracy = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
        types = typedex.keys()
        move_types = [float(t in details["type"]) for t in types]
        return [pp_frac, disabled, power, accuracy] + move_types


@dataclass
class PokemonState:
    pokemon: str
    pokemon_id: str
    level: int
    gender: str | None
    condition: str
    active: bool
    stats: dict[str, int]
    moves: list[MoveState]
    alt_moves: list[MoveState]
    transformed: bool
    was_illusion: bool
    from_enemy: bool

    @staticmethod
    def get_id(pokemon: str) -> str:
        return re.sub(r"[\s\-\.\:\’]+", "", pokemon).lower()

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

    def get_moves(self):
        if self.transformed:
            return [move.move for move in self.alt_moves]
        else:
            return [move.move for move in self.moves]

    @classmethod
    def from_request(cls, pokemon_json: Any) -> PokemonState:
        self = cls("", "", 0, None, "", False, {}, [], [], False, False, False)
        self.pokemon = pokemon_json["ident"][4:]
        self.pokemon_id = PokemonState.get_id(self.pokemon)
        level, gender = PokemonState.parse_details(pokemon_json["details"])
        self.level = level
        self.gender = gender
        self.condition = pokemon_json["condition"]
        self.active = pokemon_json["active"]
        self.stats = pokemon_json["stats"]
        self.moves = [MoveState.from_name(move) for move in pokemon_json["moves"]]
        return self

    @classmethod
    def from_protocol(cls, pokemon: str, details: str) -> PokemonState:
        pokemon_id = PokemonState.get_id(pokemon)
        level, gender = PokemonState.parse_details(details)
        return cls(
            pokemon=pokemon,
            pokemon_id=pokemon_id,
            level=level,
            gender=gender,
            condition="100/100",
            active=False,
            stats=pokedex[pokemon_id]["baseStats"],
            moves=[],
            alt_moves=[],
            transformed=False,
            was_illusion=False,
            from_enemy=True,
        )

    def switch_in(self, condition: str):
        if self.condition == "0 fnt":
            raise RuntimeError("Cannot switch in if fainted.")
        else:
            self.condition = condition
            self.active = True

    def switch_out(self):
        self.active = False
        self.alt_moves = []
        self.transformed = False

    def use_move(self, move: str, source: str | None):
        move = (
            move
            if (self.from_enemy or move != "Hidden Power")
            else [m for m in self.get_moves() if m[:12] == "Hidden Power"][0]
        )
        if (source is not None and "[from]" in source) or move == "Struggle":
            pass
        elif move in self.get_moves():
            i = self.get_moves().index(move)
            if self.transformed:
                self.alt_moves[i].pp -= 1
            else:
                self.moves[i].pp -= 1
        elif self.from_enemy:
            new_move = MoveState.from_name(move)
            new_move.pp -= 1
            if self.transformed:
                self.alt_moves.append(new_move)
            else:
                self.moves.append(new_move)
        else:
            raise RuntimeError(f"Chosen move {move} is not in pokemon {self.pokemon}'s moveset: {self.get_moves()}.")

    def faint(self):
        self.condition = "0 fnt"

    def update_condition(self, condition: str):
        self.condition = condition

    def add_status(self, status: str, reason: str | None):
        if " " not in self.condition:
            self.condition += f" {status}"
        elif reason is not None and "[from]" in reason:
            split_condition = self.condition.split(" ")
            self.condition = split_condition[0] + f" {status}"
        else:
            raise RuntimeError(f"Cannot add status {status} to condition {self.condition} of pokemon {self.pokemon}.")

    def remove_status(self, status: str):
        split_condition = self.condition.split(" ")
        if len(split_condition) > 1 and split_condition[1] == status:
            self.condition = split_condition[0]
        else:
            raise RuntimeError(
                f"Cannot remove status {status} from condition {self.condition} of pokemon {self.pokemon}."
            )

    def transform(self):
        self.transformed = True

    def process(self) -> list[float]:
        details = pokedex[self.pokemon_id]
        gender_features = [
            float(gender_bool) for gender_bool in [self.gender == "M", self.gender == "F", self.gender == None]
        ]
        condition_features = self.__process_condition(self.condition)
        stats = [stat / 255 if self.from_enemy else stat / 1000 for stat in self.stats.values()]
        types = typedex.keys()
        type_features = [float(t in details["types"]) for t in types]
        move_feature_lists = [move.process() for move in self.moves]
        move_feature_lists.extend([[0.0] * 22] * (4 - len(move_feature_lists)))
        move_features = reduce(concat_features, move_feature_lists)
        alt_move_feature_lists = [move.process() for move in self.alt_moves]
        alt_move_feature_lists.extend([[0.0] * 22] * (4 - len(alt_move_feature_lists)))
        alt_move_features = reduce(concat_features, alt_move_feature_lists)
        return gender_features + condition_features + stats + type_features + move_features + alt_move_features

    def __process_condition(self, condition: str) -> list[float]:
        if condition == "0 fnt":
            if self.from_enemy:
                return [0] * 7
            else:
                return [0] * 8
        elif " " in condition:
            hp_frac, status = condition.split(" ")
        else:
            hp_frac = condition
            status = None
        if self.from_enemy:
            numer, denom = map(float, hp_frac.split("/"))
            hp_features = [numer / denom]
        else:
            hp_left, max_hp = map(float, hp_frac.split("/"))
            hp_features = [hp_left / max_hp, max_hp / 1000]
        status_conditions = ["psn", "tox", "par", "slp", "brn", "frz"]
        status_features = [float(status == status_condition) for status_condition in status_conditions]
        return hp_features + status_features


@dataclass
class TeamState:
    __ident: str
    __is_opponent: bool
    __team: list[PokemonState]

    def __init__(self, ident: str, protocol: list[str], request: Any | None = None, is_opponent: bool = False):
        self.__ident = ident
        self.__is_opponent = is_opponent
        if request:
            self.__team = [PokemonState.from_request(pokemon_json) for pokemon_json in request["side"]["pokemon"]]
        else:
            self.__team = []
            self.update_from_protocol(protocol)

    def get_names(self) -> list[str]:
        return [pokemon.pokemon for pokemon in self.__team]

    def get_active_pokemon(self) -> PokemonState | None:
        actives = [mon for mon in self.__team if mon.active == True]
        if len(actives) == 0:
            return None
        if len(actives) == 1:
            return actives[0]
        else:
            raise RuntimeError(f"Multiple active pokemon: {[active.pokemon for active in actives]}")

    def update_from_json(self, request: Any):
        if not self.__is_opponent:
            new_pokemon_info = [PokemonState.from_request(pokemon_json) for pokemon_json in request["side"]["pokemon"]]
            for pokemon in self.__team:
                matching_info = [new_info for new_info in new_pokemon_info if new_info.pokemon == pokemon.pokemon][0]
                pokemon.condition = matching_info.condition
                pokemon.active = matching_info.active
                if pokemon.get_moves() != matching_info.get_moves() and not pokemon.alt_moves:
                    pokemon.alt_moves = matching_info.moves

    def update_from_protocol(self, protocol: list[str]):
        protocol_lines = "|".join(protocol).split("\n")
        for line in protocol_lines:
            split_line = line.split("|")
            active_pokemon = self.get_active_pokemon()
            if len(split_line) <= 2:
                pass
            elif split_line[1] in ["switch", "drag"] and self.__ident in split_line[2]:
                self.__switch(split_line[2][5:], split_line[3], split_line[4])
            elif active_pokemon and split_line[2] == f"{self.__ident}a: {active_pokemon}":
                match split_line[1]:
                    case "move":
                        active_pokemon.use_move(split_line[3], split_line[5] if len(split_line) > 5 else None)
                    case "faint":
                        active_pokemon.faint()
                    case "replace":
                        self.__replace(split_line[2][5:], split_line[3])
                    case "-damage":
                        active_pokemon.update_condition(split_line[3])
                    case "-heal":
                        active_pokemon.update_condition(split_line[3])
                    case "-status":
                        active_pokemon.add_status(split_line[3], split_line[4] if len(split_line) > 4 else None)
                    case "-curestatus":
                        active_pokemon.remove_status(split_line[3])
                    case "-transform":
                        active_pokemon.transform()
                    case _:
                        pass

    def __switch(self, pokemon: str, details: str, condition: str):
        # switch out active pokemon (if there is an active pokemon)
        active_pokemon = self.get_active_pokemon()
        if active_pokemon:
            active_pokemon.switch_out()
        # switch in desired pokemon
        if pokemon in self.get_names():
            i = self.get_names().index(pokemon)
            self.__team[i].switch_in(condition)
        else:
            new_pokemon = PokemonState.from_protocol(pokemon, details)
            new_pokemon.switch_in(condition)
            self.__team.append(new_pokemon)

    def __replace(self, pokemon: str, details: str):
        active_pokemon = self.get_active_pokemon()
        if active_pokemon:
            active_pokemon.was_illusion = True
            active_pokemon.switch_out()
            new_pokemon = PokemonState.from_protocol(pokemon, details)
            new_pokemon.switch_in(active_pokemon.condition)
            self.__team.append(new_pokemon)

    def process(self) -> list[float]:
        pokemon_feature_lists = [pokemon.process() for pokemon in self.__team]
        pokemon_feature_lists.extend([[0.0] * 210] * (6 - len(pokemon_feature_lists)))
        features = reduce(concat_features, pokemon_feature_lists)
        return features


@dataclass
class State:
    request: Any
    protocol: list[str]
    __team_state: TeamState
    __opponent_state: TeamState

    def __init__(self, protocol: list[str], request: Any):
        self.protocol = protocol
        self.request = request
        ident, enemy_ident = ("p1", "p2") if request["side"]["id"] == "p1" else ("p2", "p1")
        self.__team_state = TeamState(ident, protocol, request)
        self.__opponent_state = TeamState(enemy_ident, protocol, is_opponent=True)

    @staticmethod
    def to_dict(instance: Any) -> Any:
        if isinstance(instance, list):
            list_instance: list[Any] = instance
            return [State.to_dict(item) for item in list_instance]
        elif hasattr(instance, "__dict__"):
            return {key: State.to_dict(value) for key, value in instance.__dict__.items()}
        else:
            return instance

    def to_json(self) -> str:
        json_str = json.dumps(
            {"team_state": State.to_dict(self.__team_state), "enemy_state": State.to_dict(self.__opponent_state)},
            separators=(",", ":"),
        )
        return json_str

    def get_valid_action_ids(self) -> list[int]:
        valid_switch_ids = [
            i + 4
            for i, pokemon in enumerate(self.request["side"]["pokemon"])
            if not pokemon["active"] and pokemon["condition"] != "0 fnt"
        ]
        if "wait" in self.request:
            valid_action_ids = []
        elif "forceSwitch" in self.request:
            if "Revival Blessing" in self.protocol:
                dead_switch_ids = [
                    i + 4
                    for i, pokemon in enumerate(self.request["side"]["pokemon"])
                    if not pokemon["active"] and pokemon["condition"] == "0 fnt"
                ]
                valid_action_ids = dead_switch_ids
            else:
                valid_action_ids = valid_switch_ids
        else:
            valid_move_ids = [
                i
                for i, move in enumerate(self.request["active"][0]["moves"])
                if not ("disabled" in move and move["disabled"])
            ]
            if "trapped" in self.request["active"][0] or "maybeTrapped" in self.request["active"][0]:
                valid_action_ids = valid_move_ids
            else:
                valid_action_ids = valid_move_ids + valid_switch_ids
        return valid_action_ids

    def update(self, protocol: list[str], request: Any | None):
        self.protocol = protocol
        self.request = request
        self.__team_state.update_from_protocol(protocol)
        if request:
            self.__team_state.update_from_json(request)
        self.__opponent_state.update_from_protocol(protocol)

    def process(self) -> torch.Tensor:
        team_features = self.__team_state.process()
        enemy_features = self.__opponent_state.process()
        global_features = self.__process_globals()
        features = team_features + enemy_features + global_features
        return torch.tensor(features)

    def __process_globals(self) -> list[float]:
        gens = [f"gen{n}" for n in range(1, 10)]
        gen_features = [float(gen in self.protocol[0]) for gen in gens]
        weather_types = [
            "RainDance",
            "Sandstorm",
            "SunnyDay",
            "Snow",
            "Hail",
            "PrimordialSea",
            "DesolateLand",
            "DeltaStream",
            "none",
        ]
        if "-weather" in self.protocol:
            weather = self.protocol[self.protocol.index("-weather") + 1]
        else:
            weather = None
        weather_features = [float(weather == weather_type) for weather_type in weather_types]
        return gen_features + weather_features
