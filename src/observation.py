import json
import re
from copy import deepcopy
from functools import reduce
from typing import Any

import torch

from dex import movedex, pokedex, typedex


class Observation:
    request: Any
    protocol: list[str]
    opponent_info: Any

    def __init__(self, request: Any, protocol: list[str], past_opponent_info: Any = None):
        self.request = request
        self.protocol = protocol
        if past_opponent_info:
            self.opponent_info = self.__update_opponent_info(past_opponent_info)
        else:
            opponent_id = "p2" if request["side"]["id"] == "p1" else "p1"
            past_opponent_info = {"active": [{"moves": []}], "side": {"id": opponent_id, "pokemon": []}}
            self.opponent_info = self.__update_opponent_info(past_opponent_info)

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

    def process(self) -> torch.Tensor:
        active_features = self.__process_active(deepcopy(self.request))
        side_features = self.__process_side(deepcopy(self.request))
        global_features = self.__process_globals()
        opponent_active_features = self.__process_active(deepcopy(self.opponent_info))
        opponent_side_features = self.__process_side(deepcopy(self.opponent_info))
        features = active_features + side_features + opponent_active_features + opponent_side_features + global_features
        return torch.tensor(features)

    def __process_active(self, info: Any) -> list[float]:
        if "active" not in info:
            return [0.0] * 8
        else:
            active_moves = info["active"][0]["moves"]
            active_move_ids = [move["id"] for move in active_moves]
            active_pokemon = [mon for mon in info["side"]["pokemon"] if mon["active"] == True][0]
            i = info["side"]["pokemon"].index(active_pokemon)
            all_moves = info["side"]["pokemon"][i]["moves"]
            filtered_move_ids = [
                all_moves[i] if (i < len(all_moves) and all_moves[i] in active_move_ids) else None for i in range(4)
            ]
            filtered_moves = [
                active_moves[active_move_ids.index(move)] if move in active_move_ids else None
                for move in filtered_move_ids
            ]
            active_feature_lists = [
                [move["pp"] / move["maxpp"], float(move["disabled"])] if (move and "pp" in move) else [0.0, 0.0]
                for move in filtered_moves
            ]
            active_features = reduce(Observation.__concat_features, active_feature_lists)
            return active_features

    def __process_side(self, info: Any) -> list[float]:
        team = info["side"]["pokemon"]
        team.extend([None] * (6 - len(team)))
        from_opponent = self.request["side"]["id"] != info["side"]["id"]
        pokemon_feature_lists = [self.__process_pokemon(pokemon, from_opponent) for pokemon in team]
        side_features = reduce(Observation.__concat_features, pokemon_feature_lists)
        return side_features

    def __process_pokemon(self, pokemon: Any, from_opponent: bool) -> list[float]:
        if not pokemon:
            return [0.0] * 111
        else:
            name = re.sub(r"[\-\.\:\’\s]+", "", pokemon["ident"][4:]).lower()
            details = pokedex[name]
            condition_features = self.__process_condition(pokemon["condition"], from_opponent)
            stats = (
                [stat / 1000 for stat in pokemon["stats"].values()]
                if "stats" in pokemon
                else [stat / 255 for stat in pokemon["baseStats"].values()]
            )
            types = typedex.keys()
            type_features = [float(t in details["types"]) for t in types]
            moves = pokemon["moves"]
            moves.extend([None] * (4 - len(moves)))
            move_feature_lists = [self.__process_move(move) for move in moves]
            move_features = reduce(Observation.__concat_features, move_feature_lists)
            return condition_features + stats + type_features + move_features

    def __process_move(self, move: str) -> list[float]:
        if not move:
            return [0.0] * 20
        else:
            formatted_move = re.sub(r"\d+$", "", move)
            details = movedex[formatted_move]
            power = details["basePower"] / 250
            accuracy = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
            types = typedex.keys()
            move_types = [float(t in details["type"]) for t in types]
            return [power, accuracy] + move_types

    def __process_condition(self, condition: str, from_opponent: bool) -> list[float]:
        if condition == "0 fnt":
            if from_opponent:
                return [0] * 7
            else:
                return [0] * 8
        elif " " in condition:
            hp_frac, status = condition.split(" ")
        else:
            hp_frac = condition
            status = None
        if from_opponent:
            numer, denom = map(float, hp_frac.split("/"))
            hp_features = [numer / denom]
        else:
            hp_left, max_hp = map(float, hp_frac.split("/"))
            hp_features = [hp_left / max_hp, max_hp / 1000]
        status_conditions = ["psn", "tox", "par", "slp", "brn", "frz"]
        status_features = [float(status == status_condition) for status_condition in status_conditions]
        return hp_features + status_features

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

    def __update_opponent_info(self, past_opponent_info: Any) -> Any:
        protocol_str = "|".join(self.protocol)
        opponent_id = past_opponent_info["side"]["id"]
        if f"|move|{opponent_id}" in protocol_str:
            self.__move_update(past_opponent_info, protocol_str.split("\n"))
        if f"|switch|{opponent_id}" in protocol_str:
            self.__switch_update(past_opponent_info, protocol_str.split("\n"))
        if f"|faint|{opponent_id}" in protocol_str:
            active_pokemon = [mon for mon in past_opponent_info["side"]["pokemon"] if mon["active"] == True][0]
            i = past_opponent_info["side"]["pokemon"].index(active_pokemon)
            past_opponent_info["side"]["pokemon"][i]["condition"] = "0 fnt"
        return past_opponent_info

    def __move_update(self, past_opponent_info: Any, protocol_lines: list[str]):
        opponent_id = past_opponent_info["side"]["id"]
        line = [l for l in protocol_lines if f"|move|{opponent_id}" in l][0]
        split_line = line.split("|")
        move = split_line[3]
        if move == "Struggle":
            move_id = re.sub(r"[\-\s]+", "", move).lower()
            current_moves = [move["id"] for move in past_opponent_info["active"][0]["moves"]]
            if move_id in current_moves:
                j = current_moves.index(move_id)
                past_opponent_info["active"][0]["moves"][j]["pp"] -= 1
            else:
                details = movedex[move_id]
                past_opponent_info["active"][0]["moves"].append(
                    {
                        "id": move_id,
                        "pp": 1.6 * details["pp"] - 1 if details["pp"] > 1 else 0,
                        "maxpp": 1.6 * details["pp"],
                        "target": details["target"],
                        "disabled": False,
                    }
                )
                active_pokemon = [mon for mon in past_opponent_info["side"]["pokemon"] if mon["active"] == True][0]
                j = past_opponent_info["side"]["pokemon"].index(active_pokemon)
                past_opponent_info["side"]["pokemon"][j]["moves"].append(move_id)

    def __switch_update(self, past_opponent_info: Any, protocol_lines: list[str]):
        opponent_id = past_opponent_info["side"]["id"]

        # flipping active to false on currently active pokemon
        if past_opponent_info["side"]["pokemon"]:
            active_pokemon = [mon["ident"] for mon in past_opponent_info["side"]["pokemon"] if mon["active"] == True][0]
            i = [mon["ident"] for mon in past_opponent_info["side"]["pokemon"]].index(active_pokemon)
            past_opponent_info["side"]["pokemon"][i]["active"] = False

        line = [l for l in protocol_lines if f"|switch|{opponent_id}" in l][0]
        split_line = line.split("|")
        pokemon = split_line[2][5:]
        current_pokemon = [mon["ident"][4:] for mon in past_opponent_info["side"]["pokemon"]]
        if pokemon in current_pokemon:
            j = current_pokemon.index(pokemon)
            past_opponent_info["side"]["pokemon"][j]["active"] = True
            past_opponent_info["active"][0]["moves"] = []
            for move_id in past_opponent_info["side"]["pokemon"][j]["moves"]:
                details = movedex[move_id]
                past_opponent_info["active"][0]["moves"].append(
                    {
                        "id": move_id,
                        "pp": 1.6 * details["pp"] - 1 if details["pp"] > 1 else 0,
                        "maxpp": 1.6 * details["pp"],
                        "target": details["target"],
                        "disabled": False,
                    }
                )
        else:
            pokemon_id = re.sub(r"[\-\.\:\’\s]", "", pokemon).lower()
            details = pokedex[pokemon_id]
            past_opponent_info["side"]["pokemon"].append(
                {
                    "ident": f"{opponent_id}: {pokemon}",
                    "condition": split_line[4],
                    "active": True,
                    "baseStats": details["baseStats"],
                    "moves": [],
                }
            )
            past_opponent_info["active"][0]["moves"] = []

    def get_opponent_info_str(self) -> str:
        encoder = json.encoder.JSONEncoder(separators=(",", ":"))
        json_str = encoder.encode(self.opponent_info)
        return json_str

    @staticmethod
    def __concat_features(list1: list[float], list2: list[float]) -> list[float]:
        return list1 + list2
