import json
from functools import reduce
from typing import Any

from states.move_state import MoveState
from states.pokemon_state import PokemonState


class TeamState:
    __ident: str
    __gen: int
    __team: list[PokemonState]
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

    def __switch(self, pokemon_switching_in_name: str, details: str, hp: int, status: str | None):
        # switch out active pokemon (if there is an active pokemon)
        active_pokemon = self.get_active()
        if active_pokemon is not None:
            active_pokemon.switch_out()
        # switch in desired pokemon
        if pokemon_switching_in_name in [pokemon.name for pokemon in self.__team]:
            pokemon_switching_in = [pokemon for pokemon in self.__team if pokemon.name == pokemon_switching_in_name][0]
            pokemon_switching_in.switch_in(hp, status)
        else:
            new_pokemon = PokemonState.from_protocol(pokemon_switching_in_name, details, self.__gen, self.__ident)
            new_pokemon.switch_in(hp, status)
            self.__team.append(new_pokemon)

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
        for pokemon in self.__team:
            pokemon_info = [
                new_info for new_info in request["side"]["pokemon"] if new_info["ident"][4:] == pokemon.name
            ][0]
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
            # Conducting consistency checks if illusion pokemon isn't interfering.
            elif pokemon.active == pokemon_info["active"] and not pokemon.illusion:
                assert pokemon.hp == hp
                assert pokemon.status == status
                assert pokemon.stats == pokemon_info["stats"]
                assert len(pokemon.get_moves()) <= 4
                if pokemon.active and "active" in request and "pp" in request["active"][0]["moves"][0]:
                    zmove_pp_needs_update = self.zmove_used and not self.zmove_pp_updated
                    for move, move_info in zip(pokemon.get_moves(), request["active"][0]["moves"]):
                        if zmove_pp_needs_update or pokemon.maxed or just_unmaxed or pokemon.gen <= 3:
                            move.pp = move_info["pp"]
                        else:
                            assert move.pp == move_info["pp"]
                        assert move.maxpp == move_info["maxpp"]
                        assert move.target == move_info["target"]
                        assert move.is_disabled() == move_info["disabled"]
                    if zmove_pp_needs_update:
                        self.zmove_pp_updated = True
                    assert pokemon.can_mega == ("canMegaEvo" in request["active"][0])
                    assert pokemon.can_zmove == ("canZMove" in request["active"][0])
                    assert pokemon.can_burst == ("canUltraBurst" in request["active"][0])
                    assert pokemon.can_max == ("canDynamax" in request["active"][0])
                    assert pokemon.can_tera == ("canTerastallize" in request["active"][0])
                assert pokemon.ability == pokemon_info["baseAbility"]
                if "ability" in pokemon_info:
                    assert pokemon.get_ability() == pokemon_info["ability"]
                assert pokemon.item == (pokemon_info["item"] or None)
