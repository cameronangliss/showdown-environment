import json
import re
from typing import Any

from player import Obs


class ObsParser:
    def __init__(self):
        with open("json/pokedex.json") as f:
            self.pokedex = json.load(f)
        with open("json/moves.json") as f:
            self.movedex = json.load(f)
        with open("json/typechart.json") as f:
            self.typechart = json.load(f)

    def process_obs(self, obs: Obs) -> list[float]:
        active_features = self.process_active(obs)
        side_features = self.process_side(obs.request["side"])
        protocol_features = self.process_protocol(obs)
        return active_features + side_features + protocol_features

    def process_active(self, obs: Obs) -> list[float]:
        if "active" not in obs.request:
            return [0.0] * 8
        else:
            active_moves = obs.request["active"][0]["moves"]
            active_move_ids = [move["id"] for move in active_moves]
            all_moves = obs.request["side"]["pokemon"][0]["moves"]
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
            active_features = sum(active_feature_lists, [])
            return active_features

    def process_side(self, side_obj: Any) -> list[float]:
        side_features = sum([self.process_pokemon(pokemon) for pokemon in side_obj["pokemon"]], [])
        return side_features

    def process_pokemon(self, pokemon: Any) -> list[float]:
        if not pokemon:
            return [0.0] * 111
        else:
            name = re.sub(r"[\-\.\:\’\s]+", "", pokemon["ident"][4:]).lower()
            details = self.pokedex[name]
            condition_features = self.process_condition(pokemon["condition"])
            stats = [stat / 1000 for stat in pokemon["stats"].values()]
            types = self.typechart.keys()
            pokemon_types = [float(t in details["types"]) for t in types]
            moves = pokemon["moves"]
            moves.extend([None] * (4 - len(moves)))
            move_features = sum([self.process_move(move) for move in moves], [])
            return condition_features + stats + pokemon_types + move_features

    def process_move(self, move: str) -> list[float]:
        if not move:
            return [0.0] * 20
        else:
            formatted_move = re.sub(r"\d+$", "", move)
            details = self.movedex[formatted_move]
            power = details["basePower"] / 250
            accuracy = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
            types = self.typechart.keys()
            move_types = [float(t in details["type"]) for t in types]
            return [power, accuracy] + move_types

    def process_condition(self, condition: str) -> list[float]:
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

    def process_protocol(self, obs: Obs) -> list[float]:
        gens = [f"gen{n}" for n in range(1, 10)]
        gen_features = [float(gen in obs.protocol[0]) for gen in gens]
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
        if "-weather" in obs.protocol:
            weather = obs.protocol[obs.protocol.index("-weather") + 1]
        else:
            weather = None
        weather_features = [float(weather == weather_type) for weather_type in weather_types]
        types = self.typechart.keys()
        opponent_id = "p2" if obs.request["side"]["id"] == "p1" else "p1"
        opponent_info = [
            re.sub(r"[\-\.\:\’\s]+", "", msg[5:]).lower() for msg in obs.protocol if msg[:3] == f"{opponent_id}a"
        ]
        if opponent_info:
            opponent = opponent_info[0]
            opponent_details = self.pokedex[opponent]
            opponent_base_stats = [stat / 255 for stat in opponent_details["baseStats"].values()]
            opponent_types = [(1.0 if t in opponent_details["types"] else 0.0) for t in types]
        else:
            opponent_base_stats = [0.0] * 6
            opponent_types = [0.0] * 18
        return gen_features + weather_features + opponent_base_stats + opponent_types
