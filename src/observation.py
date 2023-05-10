import re
from dataclasses import dataclass
from functools import reduce
from typing import Any

from scrape_data import movedex, pokedex, typedex


@dataclass
class Observation:
    request: Any
    protocol: list[str]

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

    def process(self) -> list[float]:
        active_features = self.__process_active()
        side_features = self.__process_side()
        global_features = self.__process_globals()
        opponent_features = self.__process_opponent()
        return active_features + side_features + global_features + opponent_features

    def __process_active(self) -> list[float]:
        if "active" not in self.request:
            return [0.0] * 8
        else:
            active_moves = self.request["active"][0]["moves"]
            active_move_ids = [move["id"] for move in active_moves]
            all_moves = self.request["side"]["pokemon"][0]["moves"]
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

    def __process_side(self) -> list[float]:
        pokemon_feature_lists = [self.__process_pokemon(pokemon) for pokemon in self.request["side"]["pokemon"]]
        side_features = reduce(Observation.__concat_features, pokemon_feature_lists)
        return side_features

    def __process_pokemon(self, pokemon: Any) -> list[float]:
        if not pokemon:
            return [0.0] * 111
        else:
            name = re.sub(r"[\-\.\:\’\s]+", "", pokemon["ident"][4:]).lower()
            details = pokedex[name]
            condition_features = self.__process_condition(pokemon["condition"])
            stats = [stat / 1000 for stat in pokemon["stats"].values()]
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

    def __process_condition(self, condition: str) -> list[float]:
        if condition == "0 fnt":
            return [0] * 8
        elif " " in condition:
            hp_frac, status = condition.split(" ")
        else:
            hp_frac = condition
            status = None
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

    def __process_opponent(self) -> list[float]:
        types = typedex.keys()
        opponent_id = "p2" if self.request["side"]["id"] == "p1" else "p1"
        opponent_info = [
            re.sub(r"[\-\.\:\’\s]+", "", msg[5:]).lower() for msg in self.protocol if msg[:3] == f"{opponent_id}a"
        ]
        if opponent_info:
            opponent = opponent_info[0]
            opponent_details = pokedex[opponent]
            opponent_base_stats = [stat / 255 for stat in opponent_details["baseStats"].values()]
            opponent_types = [(1.0 if t in opponent_details["types"] else 0.0) for t in types]
        else:
            opponent_base_stats = [0.0] * 6
            opponent_types = [0.0] * 18
        return opponent_base_stats + opponent_types

    @staticmethod
    def __concat_features(list1: list[float], list2: list[float]) -> list[float]:
        return list1 + list2
