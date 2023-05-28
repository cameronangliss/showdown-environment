from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from states.move_state import MoveState
from states.team_state import TeamState


@dataclass
class State:
    request: Any
    protocol: list[str]
    __team_state: TeamState
    __opponent_state: TeamState
    gen: int

    def __init__(self, protocol: list[str], request: Any):
        self.protocol = protocol
        self.request = request
        ident, enemy_ident = ("p1", "p2") if request["side"]["id"] == "p1" else ("p2", "p1")
        self.gen = State.get_gen(protocol)
        self.__team_state = TeamState(ident, self.gen, protocol, request)
        self.__opponent_state = TeamState(enemy_ident, self.gen, protocol)

    @staticmethod
    def get_gen(protocol: list[str]) -> int:
        i = protocol.index("gen")
        gen = int(protocol[i + 1].strip())
        return gen

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
            i
            for i, pokemon in enumerate(self.request["side"]["pokemon"])
            if not pokemon["active"] and pokemon["condition"] != "0 fnt"
        ]
        if "wait" in self.request:
            valid_action_ids = []
        elif "forceSwitch" in self.request:
            if "Revival Blessing" in self.protocol:
                dead_switch_ids = [
                    i
                    for i, pokemon in enumerate(self.request["side"]["pokemon"])
                    if not pokemon["active"] and pokemon["condition"] == "0 fnt"
                ]
                valid_action_ids = dead_switch_ids
            else:
                valid_action_ids = valid_switch_ids
        else:
            valid_move_ids = [
                i + 6
                for i, move in enumerate(self.request["active"][0]["moves"])
                if not ("disabled" in move and move["disabled"])
            ]
            active_pokemon = self.__team_state.get_active()
            if active_pokemon:
                valid_mega_ids = (
                    [i + 4 for i in valid_move_ids]
                    if not self.__team_state.mega_used and active_pokemon.can_mega_evo
                    else []
                )
                valid_zmove_ids = [
                    i + 8
                    for i in valid_move_ids
                    if not self.__team_state.zmove_used
                    and active_pokemon.can_zmove
                    and MoveState.from_request(self.request["active"][0]["moves"][i - 6], self.gen).can_zmove(
                        active_pokemon.get_item() if active_pokemon else None
                    )
                ]
                valid_max_ids = (
                    [i + 12 for i in valid_move_ids]
                    if not self.__team_state.max_used and active_pokemon.can_max
                    else []
                )
                valid_special_ids = valid_mega_ids + valid_zmove_ids + valid_max_ids
            else:
                valid_special_ids = []
            if "trapped" in self.request["active"][0] or "maybeTrapped" in self.request["active"][0]:
                valid_action_ids = valid_move_ids + valid_special_ids
            else:
                valid_action_ids = valid_switch_ids + valid_move_ids + valid_special_ids
        return valid_action_ids

    def update(self, protocol: list[str], request: Any | None):
        self.protocol = protocol
        self.request = request
        self.__team_state.update(protocol, request)
        self.__opponent_state.update(protocol)

    def process(self) -> list[float]:
        team_features = self.__team_state.process()
        enemy_features = self.__opponent_state.process()
        global_features = self.__process_globals()
        features = team_features + enemy_features + global_features
        return features

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
