from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import reduce
from typing import Any

import torch

from dex import movedex, pokedex, typedex


@dataclass
class MoveState:
    name: str
    identifier: str
    gen: int
    pp: int
    maxpp: int
    category: str
    target: str
    just_used: bool
    disabled: bool
    disable_disabled: bool
    encore_disabled: bool
    taunt_disabled: bool
    item_disabled: bool
    no_item_disabled: bool

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
            or self.pp == 0
        )

    @classmethod
    def from_request(cls, move_json: Any, gen: int) -> MoveState:
        return cls(
            name=move_json["move"],
            identifier=move_json["id"],
            gen=gen,
            pp=move_json["pp"],
            maxpp=move_json["maxpp"],
            category=movedex[f"gen{gen}"][move_json["id"]],
            target=move_json["target"],
            just_used=False,
            disabled=move_json["disabled"],
            disable_disabled=False,
            encore_disabled=False,
            taunt_disabled=False,
            item_disabled=False,
            no_item_disabled=False,
        )

    @classmethod
    def from_name(cls, name: str, gen: int, from_mimic: bool = False) -> MoveState:
        identifier = MoveState.get_identifier(name)
        details = movedex[f"gen{gen}"][identifier]
        if from_mimic:
            pp = details["pp"]
        elif gen == 1 or gen == 2:
            pp = min(int(1.6 * details["pp"]), 61) if details["pp"] > 1 else 1
        else:
            pp = int(1.6 * details["pp"]) if details["pp"] > 1 else 1
        return cls(
            name=details["name"],
            identifier=identifier,
            gen=gen,
            pp=pp,
            maxpp=pp,
            category=details["category"],
            target=details["target"],
            just_used=False,
            disabled=False,
            disable_disabled=False,
            encore_disabled=False,
            taunt_disabled=False,
            item_disabled=False,
            no_item_disabled=False,
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
        self = cls("", "", gen, owner, 0, None, 0, 0, None, False, {}, [], [], "", None, None, False, False, False)
        self.name = pokemon_json["ident"][4:]
        self.identifier = PokemonState.get_identifier(self.name)
        level, gender = PokemonState.parse_details(pokemon_json["details"])
        self.level = level
        self.gender = gender
        split_condition = pokemon_json["condition"].split(" ")
        split_hp_frac = split_condition[0].split("/")
        self.hp = int(split_hp_frac[0])
        self.max_hp = int(split_hp_frac[1]) if pokemon_json["condition"] != "0 fnt" else 100
        self.status = split_condition[1] if len(split_condition) == 2 else None
        self.active = pokemon_json["active"]
        self.stats = pokemon_json["stats"]
        self.moves = [MoveState.from_name(name, gen) for name in pokemon_json["moves"]]
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
            new_move = MoveState.from_name(name, self.gen)
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

    def prepare(self, name: str):
        move = [move for move in self.get_moves() if move.name == name][0]
        move.pp += 1

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
            new_move = MoveState.from_name(info[1], self.gen, from_mimic=True)
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
            move = [move for move in self.get_moves() if move.name == info[1]][0]
            move.disable_disabled = False
        elif info[0] == "Encore":
            for move in self.get_moves():
                move.encore_disabled = False
        elif info[0] == "move: Taunt":
            for move in self.get_moves():
                move.taunt_disabled = False
        self.update_moves_disabled()

    def process(self) -> list[float]:
        details = pokedex[f"gen{self.gen}"][self.identifier]
        gender_features = [
            float(gender_bool) for gender_bool in [self.gender == "M", self.gender == "F", self.gender == None]
        ]
        hp_features = [self.hp / self.max_hp] if self.from_enemy else [self.hp / self.max_hp, self.max_hp / 1000]
        status_conditions = ["psn", "tox", "par", "slp", "brn", "frz", "fnt"]
        status_features = [float(self.status == status_condition) for status_condition in status_conditions]
        stats = [stat / 255 if self.from_enemy else stat / 1000 for stat in self.stats.values()]
        types = typedex[f"gen{self.gen}"].keys()
        pokemon_types = [t.lower() for t in details["types"]]
        type_features = [float(t in pokemon_types) for t in types]
        move_feature_lists = [move.process() for move in self.moves]
        move_feature_lists.extend([[0.0] * 22] * (4 - len(move_feature_lists)))
        move_features = reduce(PokemonState.concat_features, move_feature_lists)
        alt_move_feature_lists = [move.process() for move in self.alt_moves]
        alt_move_feature_lists.extend([[0.0] * 22] * (4 - len(alt_move_feature_lists)))
        alt_move_features = reduce(PokemonState.concat_features, alt_move_feature_lists)
        features = (
            gender_features + hp_features + status_features + stats + type_features + move_features + alt_move_features
        )
        return features


@dataclass
class TeamState:
    __ident: str
    __is_opponent: bool
    gen: int
    __team: list[PokemonState]
    pressure: bool = False

    def __init__(self, ident: str, protocol: list[str], request: Any | None = None, is_opponent: bool = False):
        self.__ident = ident
        self.__is_opponent = is_opponent
        self.gen = TeamState.get_gen(protocol)
        if request:
            self.__team = [
                PokemonState.from_request(pokemon_json, self.gen, self.__ident)
                for pokemon_json in request["side"]["pokemon"]
            ]
        else:
            self.__team = []
            self.update_from_protocol(protocol)

    @staticmethod
    def get_gen(protocol: list[str]) -> int:
        i = protocol.index("gen")
        gen = int(protocol[i + 1].strip())
        return gen

    def get_names(self) -> list[str]:
        return [pokemon.name for pokemon in self.__team]

    def get_active(self) -> PokemonState | None:
        actives = [mon for mon in self.__team if mon.active == True]
        if len(actives) == 0:
            return None
        if len(actives) == 1:
            return actives[0]
        else:
            raise RuntimeError(f"Multiple active pokemon: {[active.name for active in actives]}")

    def update_from_json(self, request: Any):
        if not self.__is_opponent:
            new_team_info = [
                PokemonState.from_request(pokemon_json, self.gen, self.__ident)
                for pokemon_json in request["side"]["pokemon"]
            ]
            for pokemon in self.__team:
                matching_info = [new_info for new_info in new_team_info if new_info.name == pokemon.name][0]
                # Providing new move info to a newly-transformed pokemon.
                if pokemon.transformed and not pokemon.alt_moves:
                    pokemon.alt_moves = [
                        MoveState.from_request(move_json, self.gen) for move_json in request["active"][0]["moves"]
                    ]
                # If active has been mistracked, we assume that the active pokemon is using an illusion.
                if pokemon.illusion or pokemon.active != matching_info.active:
                    pokemon.hp = matching_info.hp
                    pokemon.status = matching_info.status
                    pokemon.active = matching_info.active
                    pokemon.illusion = True
                # Conducting harsh consistency checks if illusion pokemon isn't interfering.
                else:
                    if pokemon.hp != matching_info.hp:
                        raise RuntimeError(
                            f"Mismatch of request and records. Recorded {pokemon.name} to have hp = {pokemon.hp}, but it has hp = {matching_info.hp}."
                        )
                    elif pokemon.status != matching_info.status:
                        raise RuntimeError(
                            f"Mismatch of request and records. Recorded {pokemon.name} to have status = {pokemon.status}, but it has status = {matching_info.status}."
                        )
                    elif pokemon.item != matching_info.item:
                        raise RuntimeError(
                            f"Mismatch of request and records. Recorded {pokemon.name} to have item = {pokemon.item}, but it has item = {matching_info.item}."
                        )
                    elif pokemon.active and "active" in request and "pp" in request["active"][0]["moves"][0]:
                        for move_info in request["active"][0]["moves"]:
                            move_list = [
                                move
                                for move in pokemon.get_moves()
                                if move.identifier == MoveState.get_identifier(move_info["move"])
                            ]
                            if len(move_list) > 0:
                                move = move_list[0]
                            else:
                                raise RuntimeError(
                                    f"Mismatch of request and records. Pokemon {pokemon.name} has move {MoveState.get_identifier(move_info['move'])} that isn't recorded in moveset {[move.identifier for move in pokemon.get_moves()]}."
                                )
                            if move.pp != move_info["pp"]:
                                raise RuntimeError(
                                    f"Mismatch of request and records. Recorded move {move.name} of pokemon {pokemon.name} to have pp = {move.pp}, but it has pp = {move_info['pp']}."
                                )
                            elif move.disabled != move_info["disabled"]:
                                raise RuntimeError(
                                    f"Mismatch of request and records. Recorded move {move.name} of pokemon {pokemon.name} to have disabled = {move.disabled}, but it has disabled = {move_info['disabled']}."
                                )

    def update_from_protocol(self, protocol: list[str]):
        protocol_lines = "|".join(protocol).split("\n")
        for line in protocol_lines:
            split_line = line.split("|")
            active_pokemon = self.get_active()
            if len(split_line) <= 2:
                pass
            elif split_line[1] in ["switch", "drag"] and split_line[2][:2] == self.__ident:
                hp, status = PokemonState.parse_condition(split_line[4])
                self.__switch(split_line[2][5:], split_line[3], hp, status)
            elif active_pokemon is not None and split_line[2][:2] == self.__ident:
                match split_line[1]:
                    case "move":
                        message = split_line[5] if len(split_line) > 5 else None
                        active_pokemon.use_move(split_line[3], message, self.pressure)
                    case "faint":
                        active_pokemon.update_condition(0, "fnt")
                    case "replace":
                        self.__replace(split_line[2][5:], split_line[3])
                    case "-damage":
                        hp, status = PokemonState.parse_condition(split_line[3])
                        active_pokemon.update_condition(hp, status)
                    case "-heal":
                        healed_pokemon_name = split_line[2][split_line[2].index(" ") + 1 :]
                        healed_pokemon = [pokemon for pokemon in self.__team if pokemon.name == healed_pokemon_name][0]
                        hp, status = PokemonState.parse_condition(split_line[3])
                        healed_pokemon.update_condition(hp, status)
                    case "-sethp":
                        hp, status = PokemonState.parse_condition(split_line[3])
                        active_pokemon.update_condition(hp, status)
                    case "-status":
                        active_pokemon.update_condition(active_pokemon.hp, split_line[3])
                    case "-curestatus":
                        cured_pokemon_name = split_line[2][split_line[2].index(" ") + 1 :]
                        cured_pokemon = [pokemon for pokemon in self.__team if pokemon.name == cured_pokemon_name][0]
                        cured_pokemon.update_condition(cured_pokemon.hp, None)
                    case "-cureteam":
                        for pokemon in self.__team:
                            if pokemon.status != "fnt":
                                pokemon.update_condition(pokemon.hp, None)
                    case "-ability":
                        active_pokemon.update_ability(split_line[3])
                    case "-item":
                        active_pokemon.update_item(split_line[3])
                    case "-enditem":
                        active_pokemon.update_item(None)
                    case "-prepare":
                        active_pokemon.prepare(split_line[3])
                    case "-transform":
                        active_pokemon.transform()
                    case "-start":
                        active_pokemon.start(split_line[3:])
                    case "-activate":
                        active_pokemon.start(split_line[3:])
                    case "-end":
                        active_pokemon.end(split_line[3:])
                    case _:
                        pass
            elif split_line[2][:2] != self.__ident:
                if split_line[1] in ["switch", "drag"]:
                    if self.pressure:
                        print(f"Player {self.__ident} sensed the pressure ease.")
                        self.pressure = False
                elif split_line[1] == "-ability" and split_line[3] == "Pressure":
                    print(f"Player {self.__ident} sensed pressure from {split_line[2]}.")
                    self.pressure = True
                elif split_line[1] == "-item" and len(split_line) > 4 and split_line[4] == "[from] move: Thief":
                    if active_pokemon is not None:
                        active_pokemon.update_item(None)
                    else:
                        raise RuntimeError("A pokemon must be active for its item to be stolen.")

    def __switch(self, pokemon: str, details: str, hp: int, status: str | None):
        # switch out active pokemon (if there is an active pokemon)
        active_pokemon = self.get_active()
        if active_pokemon:
            active_pokemon.switch_out()
        # switch in desired pokemon
        if pokemon in self.get_names():
            i = self.get_names().index(pokemon)
            self.__team[i].switch_in(hp, status)
        else:
            new_pokemon = PokemonState.from_protocol(pokemon, details, self.gen, self.__ident)
            new_pokemon.switch_in(hp, status)
            self.__team.append(new_pokemon)

    def __replace(self, pokemon: str, details: str):
        active_pokemon = self.get_active()
        if active_pokemon:
            actual_pokemon = PokemonState.from_protocol(pokemon, details, self.gen, self.__ident)
            actual_pokemon.switch_in(active_pokemon.hp, active_pokemon.status)
            active_pokemon = actual_pokemon
        else:
            raise RuntimeError("Cannot replace pokemon if there are no active pokemon.")

    def process(self) -> list[float]:
        pokemon_feature_lists = [pokemon.process() for pokemon in self.__team]
        pokemon_feature_lists.extend([[0.0] * 211] * (6 - len(pokemon_feature_lists)))
        features = reduce(PokemonState.concat_features, pokemon_feature_lists)
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
        gen_features = [float(n == self.__team_state.gen) for n in range(1, 10)]
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
