import json
from functools import reduce
from typing import Any

from states.move_state import MoveState
from states.pokemon_state import PokemonState


class TeamState:
    __ident: str
    __gen: int
    __team: list[PokemonState]
    opponent_team: list[PokemonState]
    __pressure: bool = False
    mega_used: bool = False
    zmove_used: bool = False
    zmove_pp_updated: bool = False
    burst_used: bool = False
    max_used: bool = False
    tera_used: bool = False

    def __init__(self, ident: str, gen: int, protocol: list[str], request: Any | None = None):
        self.__ident = ident
        self.__gen = gen
        self.__team = (
            [
                PokemonState.from_request(pokemon_json, self.__gen, self.__ident)
                for pokemon_json in request["side"]["pokemon"]
            ]
            if request
            else []
        )
        self.update(protocol, request)

    ###################################################################################################################
    # Getter methods

    def get_active(self) -> PokemonState | None:
        actives = [mon for mon in self.__team if mon.active]
        if len(actives) == 0:
            return None
        if len(actives) == 1:
            return actives[0]
        else:
            raise RuntimeError(f"Multiple active pokemon: {[active.name for active in actives]}")

    def get_json_str(self) -> str:
        return json.dumps([json.loads(pokemon.get_json_str()) for pokemon in self.__team])

    ###################################################################################################################
    # Self-updating methods used when reading through the lines of the protocol and the request

    def update(self, protocol: list[str], request: Any | None = None):
        self.__update_from_protocol(protocol)
        if request:
            protocol_lines = "|".join(protocol).split("\n")
            active_pokemon = self.get_active()
            active_name = active_pokemon.name if active_pokemon else ""
            just_unmaxed = f"|-end|{self.__ident}a: {active_name}|Dynamax" in protocol_lines
            self.__update_from_request(request, just_unmaxed)
            if not any([pokemon.illusion for pokemon in self.__team]):
                self.check_consistency(request, just_unmaxed)

    def __update_from_protocol(self, protocol: list[str]):
        protocol_lines = "|".join(protocol).split("\n")
        for line in protocol_lines:
            split_line = line.split("|")
            active_pokemon = self.get_active()
            if active_pokemon is None:
                if len(split_line) > 2 and split_line[1] == "switch" and split_line[2][:2] == self.__ident:
                    hp, status = PokemonState.parse_condition(split_line[4])
                    self.__switch(split_line[2][5:], split_line[3], hp, status)
            elif len(split_line) > 2 and split_line[2][:2] == self.__ident:
                match split_line[1]:
                    case "move":
                        if active_pokemon.preparing:
                            active_pokemon.preparing = False
                        else:
                            active_pokemon.use_move(split_line[3], split_line[5:], self.__pressure)
                    case "switch":
                        hp, status = PokemonState.parse_condition(split_line[4])
                        self.__switch(split_line[2][5:], split_line[3], hp, status)
                    case "drag":
                        hp, status = PokemonState.parse_condition(split_line[4])
                        self.__switch(split_line[2][5:], split_line[3], hp, status)
                    case "faint":
                        fainted_pokemon_name = split_line[2][split_line[2].index(" ") + 1 :]
                        fainted_pokemon = [pokemon for pokemon in self.__team if pokemon.name == fainted_pokemon_name][
                            0
                        ]
                        fainted_pokemon.update_condition(0, "fnt")
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
                        active_pokemon.update_item(split_line[3], split_line[4:])
                    case "-enditem":
                        active_pokemon.end_item(split_line[3], split_line[4:])
                    case "-prepare":
                        if self.__gen <= 2:
                            active_pokemon.preparing = True
                    case "-mega":
                        self.mega_used = True
                        active_pokemon.mega_evolve(split_line[4] or None)
                    case "-primal":
                        active_pokemon.primal_reversion()
                    case "-zpower":
                        self.zmove_used = True
                    case "-burst":
                        self.burst_used = True
                    case "-terastallize":
                        self.tera_used = True
                    case "-transform":
                        copied_pokemon_name = split_line[3][split_line[3].index(" ") + 1 :]
                        active_pokemon.transform(copied_pokemon_name)
                    case "-start":
                        if len(split_line) > 3 and split_line[3] == "Dynamax":
                            self.max_used = True
                        active_pokemon.start(split_line[3:])
                    case "-activate":
                        if len(split_line) > 3:
                            if split_line[3] == "ability: Mummy" and self.__pressure:
                                self.__pressure = False
                            if split_line[3] == "Dynamax":
                                self.max_used = True
                        active_pokemon.start(split_line[3:])
                    case "-end":
                        active_pokemon.end(split_line[3:])
                    case _:
                        pass
            elif len(split_line) > 2 and split_line[2][:2] != self.__ident:
                match split_line[1]:
                    case "switch":
                        if self.__pressure:
                            self.__pressure = False
                    case "drag":
                        if self.__pressure:
                            self.__pressure = False
                    case "faint":
                        if self.__pressure:
                            self.__pressure = False
                    case "-ability":
                        if split_line[3] == "Pressure":
                            self.__pressure = True
                    case "-item":
                        if len(split_line) > 4 and split_line[4] in ["[from] move: Thief", "[from] ability: Magician"]:
                            active_pokemon.end_item(split_line[3], split_line[4:])
                    case "-activate":
                        if len(split_line) > 3 and split_line[3] == "ability: Mummy":
                            active_pokemon.alt_ability = "mummy"
                    case _:
                        pass
            if active_pokemon is not None:
                active_pokemon.update_special_options(
                    self.mega_used, self.zmove_used, self.burst_used, self.max_used, self.tera_used
                )
        active_pokemon = self.get_active()
        if active_pokemon is not None:
            active_pokemon.tricking = False

    def __switch(self, incoming_pokemon_name: str, details: str, hp: int, status: str | None):
        # get incoming pokemon
        if incoming_pokemon_name in [pokemon.name for pokemon in self.__team]:
            incoming_pokemon = [pokemon for pokemon in self.__team if pokemon.name == incoming_pokemon_name][0]
            incoming_index = self.__team.index(incoming_pokemon)
        else:
            incoming_pokemon = PokemonState.from_protocol(incoming_pokemon_name, details, self.__gen, self.__ident)
            incoming_index = None
        # get outgoing pokemon
        outgoing_pokemon = self.get_active()
        if outgoing_pokemon is None:
            outgoing_index = None if incoming_index is None else 0
        else:
            outgoing_index = self.__team.index(outgoing_pokemon)
            outgoing_pokemon.switch_out()
        incoming_pokemon.switch_in(hp, status)
        # place incoming pokemon in new position
        if outgoing_index is None:
            self.__team.append(incoming_pokemon)
        else:
            self.__team[outgoing_index] = incoming_pokemon
        # place outgoing pokemon in new position
        if outgoing_pokemon is not None:
            if incoming_index is None:
                self.__team.append(outgoing_pokemon)
            else:
                self.__team[incoming_index] = outgoing_pokemon

    def __replace(self, pokemon: str, details: str):
        active_pokemon = self.get_active()
        if active_pokemon is not None:
            # If the player didn't know about the illusion before it was revealed, corrections need to be made.
            if not active_pokemon.illusion:
                actual_pokemon = PokemonState.from_protocol(pokemon, details, self.__gen, self.__ident)
                actual_pokemon.switch_in(active_pokemon.hp, active_pokemon.status)
                active_pokemon = actual_pokemon
        else:
            raise RuntimeError("Cannot replace pokemon if there are no active pokemon.")

    def __update_from_request(self, request: Any, just_unmaxed: bool):
        team_info = [pokemon_info for pokemon_info in request["side"]["pokemon"]]
        for pokemon, pokemon_info in zip(self.__team, team_info):
            # Providing new info to a newly-transformed pokemon.
            if pokemon.transformed and pokemon.alt_stats is None:
                pokemon.alt_moves = [
                    MoveState(move_name, self.__gen, "ghost" in pokemon.get_types())
                    for move_name in pokemon_info["moves"]
                ]
                for move in pokemon.alt_moves:
                    move.pp = 5
                pokemon.alt_stats = pokemon_info["stats"]
                pokemon.alt_ability = (
                    pokemon_info["ability"] if "ability" in pokemon_info else pokemon_info["baseAbility"]
                )
            # If active has been mistracked, we assume that the active pokemon is using an illusion.
            hp, status = PokemonState.parse_condition(pokemon_info["condition"])
            if pokemon.active != pokemon_info["active"]:
                pokemon.hp = hp
                pokemon.status = status
                pokemon.active = pokemon_info["active"]
                pokemon.illusion = True

    ###################################################################################################################
    # Consistency checking

    def check_consistency(self, request: Any, just_unmaxed: bool):
        team_info = [pokemon_info for pokemon_info in request["side"]["pokemon"]]
        for pokemon, pokemon_info in zip(self.__team, team_info):
            zmove_pp_needs_update = self.zmove_used and not self.zmove_pp_updated
            active_info = request["active"] if "active" in request else None
            pokemon.check_consistency(pokemon_info, active_info, zmove_pp_needs_update, just_unmaxed)
            if zmove_pp_needs_update:
                self.zmove_pp_updated = True

    ###################################################################################################################
    # Processes TeamState object into a feature vector to be fed into the model's input layer

    def process(self) -> list[float]:
        pokemon_feature_lists = [pokemon.process() for pokemon in self.__team]
        pokemon_feature_lists.extend([[0.0] * 123] * (6 - len(pokemon_feature_lists)))
        pokemon_features = reduce(lambda features1, features2: features1 + features2, pokemon_feature_lists)
        special_used_features = [
            float(self.mega_used),
            float(self.zmove_used),
            float(self.max_used),
            float(self.tera_used),
        ]
        features = pokemon_features + special_used_features
        return features
