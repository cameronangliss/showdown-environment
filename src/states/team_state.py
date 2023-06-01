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

    def __get_names(self) -> list[str]:
        return [pokemon.name for pokemon in self.__team]

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
            if active_pokemon is not None and len(split_line) > 2 and split_line[2][:2] == self.__ident:
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
                        active_pokemon.mega_evolve()
                    case "-primal":
                        active_pokemon.primal_reversion()
                    case "-zpower":
                        self.zmove_used = True
                    case "-burst":
                        self.burst_used = True
                    case "-terastallize":
                        self.tera_used = True
                    case "-transform":
                        active_pokemon.transformed = True
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
            elif active_pokemon is not None and len(split_line) > 2 and split_line[2][:2] != self.__ident:
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
                        if len(split_line) > 4 and split_line[4] == "[from] move: Thief":
                            active_pokemon.end_item(split_line[3], split_line[4:])
                    case "-activate":
                        if len(split_line) > 3 and split_line[3] == "ability: Mummy":
                            active_pokemon.alt_ability = "mummy"
                    case _:
                        pass
            if active_pokemon:
                active_pokemon.update_special_options(
                    self.mega_used, self.zmove_used, self.burst_used, self.max_used, self.tera_used
                )

    def __switch(self, pokemon: str, details: str, hp: int, status: str | None):
        # switch out active pokemon (if there is an active pokemon)
        active_pokemon = self.get_active()
        if active_pokemon:
            active_pokemon.switch_out()
        # switch in desired pokemon
        if pokemon in self.__get_names():
            i = self.__get_names().index(pokemon)
            self.__team[i].switch_in(hp, status)
        else:
            new_pokemon = PokemonState.from_protocol(pokemon, details, self.__gen, self.__ident)
            new_pokemon.switch_in(hp, status)
            self.__team.append(new_pokemon)

    def __replace(self, pokemon: str, details: str):
        active_pokemon = self.get_active()
        if active_pokemon and not active_pokemon.illusion:
            actual_pokemon = PokemonState.from_protocol(pokemon, details, self.__gen, self.__ident)
            actual_pokemon.switch_in(active_pokemon.hp, active_pokemon.status)
            active_pokemon = actual_pokemon
        elif active_pokemon and active_pokemon.illusion:
            pass
        else:
            raise RuntimeError("Cannot replace pokemon if there are no active pokemon.")

    def __update_from_request(self, request: Any, just_unmaxed: bool):
        for pokemon in self.__team:
            pokemon_info = [
                new_info for new_info in request["side"]["pokemon"] if new_info["ident"][4:] == pokemon.name
            ][0]
            # Providing new move info to a newly-transformed pokemon.
            if pokemon.transformed and not pokemon.alt_moves and "active" in request:
                pokemon.alt_moves = [
                    MoveState.from_request(move_json, self.__gen, pokemon.item)
                    for move_json in request["active"][0]["moves"]
                ]
                pokemon.alt_ability = (
                    pokemon_info["ability"] if "ability" in pokemon_info else pokemon_info["baseAbility"]
                )
            # If active has been mistracked, we assume that the active pokemon is using an illusion.
            if pokemon.active != pokemon_info["active"]:
                hp, status = PokemonState.parse_condition(pokemon_info["condition"])
                pokemon.hp = hp
                pokemon.status = status
                pokemon.active = pokemon_info["active"]
                pokemon.illusion = True
            # Conducting harsh consistency checks if illusion pokemon isn't interfering.
            elif pokemon.active == pokemon_info["active"] and not pokemon.illusion:
                TeamState.__check_condition_consistency(pokemon, pokemon_info)
                if pokemon.active and "active" in request and "pp" in request["active"][0]["moves"][0]:
                    self.__check_moves_consistency(pokemon, request["active"], just_unmaxed)
                    TeamState.__check_can_special_consistency(pokemon, request["active"])
                TeamState.__check_ability_consistency(pokemon, pokemon_info)
                TeamState.__check_item_consistency(pokemon, pokemon_info)

    ###################################################################################################################
    # Helper functions

    @staticmethod
    def __check_condition_consistency(pokemon: PokemonState, pokemon_info: Any):
        hp, status = PokemonState.parse_condition(pokemon_info["condition"])
        if pokemon.hp != hp:
            raise RuntimeError(
                f"Mismatch of request and records. Recorded {pokemon.name} to have hp = {pokemon.hp}, but it has "
                f"hp = {hp}."
            )
        elif pokemon.status != status:
            raise RuntimeError(
                f"Mismatch of request and records. Recorded {pokemon.name} to have status = {pokemon.status}, but it "
                f"has status = {status}."
            )

    def __check_moves_consistency(self, pokemon: PokemonState, active_info: Any, just_unmaxed: bool):
        if len(pokemon.moves) > 4:
            raise RuntimeError(f"Pokemon cannot have more than 4 moves: {[move.name for move in pokemon.moves]}.")
        elif len(pokemon.alt_moves) > 4:
            raise RuntimeError(
                f"Pokemon cannot have more than 4 alt_moves: {[move.name for move in pokemon.alt_moves]}."
            )
        zmove_pp_needs_update = self.zmove_used and not self.zmove_pp_updated
        for move_info in active_info[0]["moves"]:
            move = TeamState.__get_matching_move(pokemon, move_info)
            if zmove_pp_needs_update or pokemon.maxed or just_unmaxed or pokemon.gen <= 3:
                move.pp = move_info["pp"]
            TeamState.__check_move_consistency(move, move_info)
        if zmove_pp_needs_update:
            self.zmove_pp_updated = True

    @staticmethod
    def __check_move_consistency(move: MoveState, move_info: Any):
        if move.pp != move_info["pp"]:
            raise RuntimeError(
                f"Mismatch of request and records. Recorded move {move.name} to have pp = {move.pp}, but it has "
                f"pp = {move_info['pp']}."
            )
        elif move.is_disabled() != move_info["disabled"]:
            raise RuntimeError(
                f"Mismatch of request and records. Recorded move {move.name} to have disabled = {move.is_disabled()}, "
                f"but it has disabled = {move_info['disabled']}."
            )

    @staticmethod
    def __get_matching_move(pokemon: PokemonState, move_info: Any):
        move_list = [
            move for move in pokemon.get_moves() if move.identifier == MoveState.get_identifier(move_info["move"])
        ]
        if len(move_list) > 0:
            return move_list[0]
        else:
            raise RuntimeError(
                f"Mismatch of request and records. Pokemon {pokemon.name} has move "
                f"{MoveState.get_identifier(move_info['move'])} that isn't recorded in moveset "
                f"{[move.identifier for move in pokemon.get_moves()]}."
            )

    @staticmethod
    def __check_ability_consistency(pokemon: PokemonState, pokemon_info: Any):
        if pokemon.ability != pokemon_info["baseAbility"]:
            raise RuntimeError(
                f"Mismatch of request and records. Recorded {pokemon.name} to have baseAbility = {pokemon.ability}, "
                f"but it has baseAbility = {pokemon_info['baseAbility']}."
            )
        if "ability" in pokemon_info and pokemon.get_ability() != pokemon_info["ability"]:
            raise RuntimeError(
                f"Mismatch of request and records. Recorded {pokemon.name} to have ability = {pokemon.get_ability()}, "
                f"but it has ability = {pokemon_info['ability']}."
            )

    @staticmethod
    def __check_item_consistency(pokemon: PokemonState, pokemon_info: Any):
        if pokemon.item != (pokemon_info["item"] or None):
            raise RuntimeError(
                f"Mismatch of request and records. Recorded {pokemon.name} to have item = {pokemon.item}, but it "
                f"has item = {pokemon_info['item'] or None}."
            )

    @staticmethod
    def __check_can_special_consistency(pokemon: PokemonState, active_info: Any):
        if pokemon.can_mega != ("canMegaEvo" in active_info[0]):
            raise RuntimeError(
                f"Mismatch of request and records. Recorded pokemon {pokemon.name} to have can_mega = "
                f"{pokemon.can_mega}, but it has can_mega = {'canMegaEvo' in active_info[0]}."
            )
        elif pokemon.can_zmove != ("canZMove" in active_info[0]):
            raise RuntimeError(
                f"Mismatch of request and records. Recorded pokemon {pokemon.name} to have can_zmove = "
                f"{pokemon.can_zmove}, but it has can_zmove = {'canZMove' in active_info[0]}."
            )
        elif pokemon.can_burst != ("canUltraBurst" in active_info[0]):
            raise RuntimeError(
                f"Mismatch of request and records. Recorded pokemon {pokemon.name} to have can_burst = "
                f"{pokemon.can_burst}, but it has can_burst = {'canUltraBurst' in active_info[0]}."
            )

        elif pokemon.can_max != ("canDynamax" in active_info[0]):
            raise RuntimeError(
                f"Mismatch of request and records. Recorded pokemon {pokemon.name} to have can_max = "
                f"{pokemon.can_max}, but it has can_max = {'canDynamax' in active_info[0]}."
            )
        elif pokemon.can_tera != ("canTerastallize" in active_info[0]):
            raise RuntimeError(
                f"Mismatch of request and records. Recorded pokemon {pokemon.name} to have can_tera = "
                f"{pokemon.can_tera}, but it has can_tera = {'canTerastallize' in active_info[0]}."
            )
