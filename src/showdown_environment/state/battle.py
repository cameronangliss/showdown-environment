from __future__ import annotations

import json
from typing import Any

from showdown_environment.data.dex import movedex
from showdown_environment.state.move import Move
from showdown_environment.state.team import Team


class Battle:
    request: Any
    protocol: list[str]
    gen: int
    team: Team
    opponent_team: Team

    def __init__(self, protocol: list[str], request: Any):
        self.protocol = protocol
        self.request = request
        ident, opponent_ident = ("p1", "p2") if request["side"]["id"] == "p1" else ("p2", "p1")
        self.gen = self.__get_gen()
        self.team = Team(ident, self.gen, protocol, request)
        self.opponent_team = Team(opponent_ident, self.gen, protocol)

    def update(self, protocol: list[str], request: Any | None):
        self.protocol = protocol
        self.request = request
        self.team.update(protocol, request)
        self.opponent_team.update(protocol)

    def update_in_simulation(self, action: int, opp_action: int):
        team = [
            ps_sim.Pokemon(
                p.name,
                p.level,
                [m.name for m in p.get_moves()],
                p.gender or "",
                ability=p.ability,
                cur_hp=p.hp,
                stats_actual=p.stats,
                item=p.get_item() or "",
                status=p.status or "",
            )
            for p in self.team.team
        ]
        opp_team = [
            ps_sim.Pokemon(
                p.name,
                p.level,
                [m.name for m in p.get_moves()],
                p.gender or "",
                ability=p.ability,
                cur_hp=p.hp,
                stats_actual=p.stats,
                item=p.get_item() or "",
                status=p.status or "",
            )
            for p in self.opponent_team.team
        ]
        me = ps_sim.Trainer("p1", team)
        opp = ps_sim.Trainer("p2", opp_team)
        battle = ps_sim.Battle(t1=me, t2=opp)
        battle.start()
        action_space = [["other", p.name] for p in self.team.team]
        battle.turn([], [])

    def infer_opponent_sets(self):
        for pokemon in self.opponent_team.team:
            matching_role = pokemon.get_matching_role()
            pokemon.ability = matching_role["abilities"][0]
            pokemon.item = matching_role["items"][0]
            new_moves = [
                Move(
                    move_name,
                    self.gen,
                    "ghost" in movedex[Move.get_identifier(move_name)]["type"],
                )
                for move_name in matching_role["moves"]
                if move_name not in pokemon.get_moves()
            ][: 4 - len(pokemon.get_moves())]
            pokemon.moves.extend(new_moves)

    ###############################################################################################
    # Getter methods

    def __get_gen(self) -> int:
        i = self.protocol.index("gen")
        gen = int(self.protocol[i + 1].strip())
        return gen

    def get_json_str(self) -> str:
        json_str = json.dumps(
            {
                "##### team_state #####": json.loads(self.team.get_json_str()),
                "##### opponent_state #####": json.loads(self.opponent_team.get_json_str()),
            },
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
            active_pokemon = self.team.get_active()
            if active_pokemon and self.gen >= 6:
                valid_mega_ids = (
                    [i + 4 for i in valid_move_ids]
                    if "canMegaEvo" in self.request["active"][0]
                    else []
                )
                valid_zmove_ids = (
                    [
                        i + 6 + 8
                        for i, move in enumerate(self.request["active"][0]["canZMove"])
                        if move is not None
                    ]
                    if "canZMove" in self.request["active"][0]
                    else (
                        [
                            i + 6 + 8
                            for i, move in self.request["active"][0]["moves"]
                            if move["move"] == "Photon Geyser"
                        ]
                        if "canUltraBurst" in self.request["active"][0]
                        else []
                    )
                )
                valid_max_ids = (
                    [i + 12 for i in valid_move_ids]
                    if "canDynamax" in self.request["active"][0]
                    else []
                )
                valid_tera_ids = (
                    [i + 16 for i in valid_move_ids]
                    if "canTerastallize" in self.request["active"][0]
                    else []
                )
                valid_special_ids = (
                    valid_mega_ids + valid_zmove_ids + valid_max_ids + valid_tera_ids
                )
            else:
                valid_special_ids = []
            if (
                "trapped" in self.request["active"][0]
                or "maybeTrapped" in self.request["active"][0]
            ):
                valid_action_ids = valid_move_ids + valid_special_ids
            else:
                valid_action_ids = valid_switch_ids + valid_move_ids + valid_special_ids
        return valid_action_ids
