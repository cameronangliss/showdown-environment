import json
import random
from subprocess import PIPE, run
from typing import Any

from showdown_environment.state.battle import Battle


class Model:
    def predict(self, state: Battle, action: str, opp_action: str) -> Battle:
        active_pokemon = state.team.get_active()
        opp_active_pokemon = state.opponent_team.get_active()
        assert active_pokemon is not None
        assert opp_active_pokemon is not None
        speed = active_pokemon.stats["spd"]
        opp_speed = self.__calc_stat(
            gen=state.gen,
            stat_id="spd",
            base=opp_active_pokemon.stats["spd"],
            iv=31,
            ev=0,
            level=opp_active_pokemon.level,
            nature=None,
        )
        first_mon, first_action = (
            (active_pokemon, action)
            if speed > opp_speed
            else (
                (opp_active_pokemon, opp_action)
                if opp_speed > speed
                else (
                    (active_pokemon, action)
                    if random.random() < 0.5
                    else (opp_active_pokemon, opp_action)
                )
            )
        )
        second_mon, second_action = (
            (opp_active_pokemon, opp_action)
            if first_mon == active_pokemon
            else (active_pokemon, action)
        )
        protocol: list[str] = []
        if "move " in action and "move " in opp_action:
            first_damage_rolls = self.__calc_damage(
                gen=state.gen,
                attacker=first_mon.name,
                attacker_info={
                    "item": first_mon.get_item(),
                    "nature": None,
                    "stats": first_mon.get_stats(),
                    "boosts": {},
                },
                defender=second_mon.name,
                defender_info={
                    "item": second_mon.get_item(),
                    "nature": None,
                    "stats": second_mon.get_stats(),
                    "boosts": {},
                },
                move=first_mon.moves[int(action[-1]) - 1].name,
            )
            second_damage_rolls = self.__calc_damage(
                gen=state.gen,
                attacker=second_mon.name,
                attacker_info={
                    "item": second_mon.get_item(),
                    "nature": None,
                    "stats": second_mon.get_stats(),
                    "boosts": {},
                },
                defender=first_mon.name,
                defender_info={
                    "item": first_mon.get_item(),
                    "nature": None,
                    "stats": first_mon.get_stats(),
                    "boosts": {},
                },
                move=second_mon.moves[int(opp_action[-1]) - 1].name,
            )
            protocol += [
                f"move | {first_mon.name} | {first_mon.moves[int(first_action[-1]) - 1]}",
                f"damage | {second_mon.name} | {first_damage_rolls}"
                f"move | {second_mon.name} | {second_mon.moves[int(second_action[-1]) - 1]}",
                f"damage | {first_mon.name} | {second_damage_rolls}",
            ]
        elif "switch " in action and "switch " in opp_action:
            protocol += [
                f"switch | {state.team.team[int(first_action[-1]) - 1].name}",
                f"switch | {state.opponent_team.team[int(second_action[-1]) - 1].name}",
            ]
        else:
            pass
        state.update(protocol, request=None)
        return state

    @staticmethod
    def __calc_damage(
        gen: int,
        attacker: str,
        attacker_info: dict[str, Any],
        defender: str,
        defender_info: dict[str, Any],
        move: str,
    ) -> list[int]:
        json_obj = {
            "gen": gen,
            "attacker": attacker,
            "attacker_info": attacker_info,
            "defender": defender,
            "defender_info": defender_info,
            "move": move,
        }
        stdin = json.dumps(json_obj)
        result = run(
            args=["node", "src/showdown_environment/js/calc_damage.js"],
            input=stdin.encode("utf8"),
            stdout=PIPE,
        )
        return json.loads(result.stdout.decode("utf8"))

    @staticmethod
    def __calc_stat(
        gen: int, stat_id: str, base: int, iv: int, ev: int, level: int, nature: str | None
    ):
        json_obj = {
            "gen": gen,
            "id": stat_id,
            "base": base,
            "iv": iv,
            "ev": ev,
            "level": level,
            "nature": nature,
        }
        stdin = json.dumps(json_obj)
        result = run(
            args=["node", "src/showdown_environment/js/calc_stat.js"],
            input=stdin.encode("utf8"),
            stdout=PIPE,
        )
        return json.loads(result.stdout.decode("utf8"))
