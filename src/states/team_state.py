from dataclasses import dataclass
from functools import reduce
from typing import Any

from states.move_state import MoveState
from states.pokemon_state import PokemonState


@dataclass
class TeamState:
    __ident: str
    gen: int
    __team: list[PokemonState]
    pressure: bool = False
    mega_used: bool = False
    zmove_used: bool = False
    max_used: bool = False

    def __init__(self, ident: str, gen: int, protocol: list[str], request: Any | None = None):
        self.__ident = ident
        self.gen = gen
        self.__team = (
            [
                PokemonState.from_request(pokemon_json, self.gen, self.__ident)
                for pokemon_json in request["side"]["pokemon"]
            ]
            if request
            else []
        )
        self.update(protocol, request)

    def get_names(self) -> list[str]:
        return [pokemon.name for pokemon in self.__team]

    def get_active(self) -> PokemonState | None:
        actives = [mon for mon in self.__team if mon.active]
        if len(actives) == 0:
            return None
        if len(actives) == 1:
            return actives[0]
        else:
            raise RuntimeError(f"Multiple active pokemon: {[active.name for active in actives]}")

    def update(self, protocol: list[str], request: Any | None = None):
        self.update_from_protocol(protocol)
        # if request:
        #     protocol_lines = "|".join(protocol).split("\n")
        #     active_pokemon = self.get_active()
        #     active_name = active_pokemon.name if active_pokemon else ""
        #     just_mega_evod = any(
        #         f"|-mega|{self.__ident}a: {active_name}|{active_name}" in line for line in protocol_lines
        #     )
        #     just_zmoved = f"|-zpower|{self.__ident}a: {active_name}" in protocol_lines
        #     just_unmaxed = f"|-end|{self.__ident}a: {active_name}|Dynamax" in protocol_lines
        #     self.update_from_request(request, just_mega_evod, just_zmoved, just_unmaxed)

    def update_from_protocol(self, protocol: list[str]):
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
                            active_pokemon.use_move(split_line[3], split_line[5:], self.pressure)
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
                        if self.gen <= 2:
                            active_pokemon.preparing = True
                    case "-mega":
                        self.mega_used = True
                    case "-zpower":
                        self.zmove_used = True
                    case "-transform":
                        active_pokemon.transform()
                    case "-start":
                        if len(split_line) > 3 and split_line[3] == "Dynamax":
                            self.max_used = True
                        active_pokemon.start(split_line[3:])
                    case "-activate":
                        if len(split_line) > 3:
                            if split_line[3] == "ability: Mummy" and self.pressure:
                                self.pressure = False
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
                        if self.pressure:
                            self.pressure = False
                    case "drag":
                        if self.pressure:
                            self.pressure = False
                    case "faint":
                        if self.pressure:
                            self.pressure = False
                    case "-ability":
                        if split_line[3] == "Pressure":
                            self.pressure = True
                    case "-item":
                        if len(split_line) > 4 and split_line[4] == "[from] move: Thief":
                            active_pokemon.end_item(split_line[3], split_line[4:])
                    case "-activate":
                        if len(split_line) > 3 and split_line[3] == "ability: Mummy":
                            active_pokemon.alt_ability = "mummy"
                    case _:
                        pass
            if active_pokemon:
                active_pokemon.update_special_options(self.mega_used, self.zmove_used, self.max_used)

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
        if active_pokemon and not active_pokemon.illusion:
            actual_pokemon = PokemonState.from_protocol(pokemon, details, self.gen, self.__ident)
            actual_pokemon.switch_in(active_pokemon.hp, active_pokemon.status)
            active_pokemon = actual_pokemon
        elif active_pokemon and active_pokemon.illusion:
            pass
        else:
            raise RuntimeError("Cannot replace pokemon if there are no active pokemon.")

    def update_from_request(self, request: Any, just_mega_evod: bool, just_zmoved: bool, just_unmaxed: bool):
        for pokemon in self.__team:
            pokemon_info = [
                new_info for new_info in request["side"]["pokemon"] if new_info["ident"][4:] == pokemon.name
            ][0]
            # Providing new move info to a newly-transformed pokemon.
            if pokemon.transformed and not pokemon.alt_moves and "active" in request:
                pokemon.alt_moves = [
                    MoveState.from_request(move_json, self.gen) for move_json in request["active"][0]["moves"]
                ]
            # If active has been mistracked, we assume that the active pokemon is using an illusion.
            if pokemon.active != pokemon_info["active"]:
                hp, status = PokemonState.parse_condition(pokemon_info["condition"])
                pokemon.hp = hp
                pokemon.status = status
                pokemon.active = pokemon_info["active"]
                pokemon.illusion = True
            # Conducting harsh consistency checks if illusion pokemon isn't interfering.
            elif pokemon.active == pokemon_info["active"] and not pokemon.illusion:
                TeamState.check_condition_consistency(pokemon, pokemon_info)
                if just_mega_evod:
                    pokemon.ability = pokemon_info["baseAbility"]
                # TeamState.check_ability_consistency(pokemon, pokemon_info)
                TeamState.check_item_consistency(pokemon, pokemon_info)
                if len(pokemon.moves) > 4:
                    raise RuntimeError(
                        f"Pokemon cannot have more than 4 moves: {[move.name for move in pokemon.moves]}."
                    )
                elif pokemon.active and "active" in request and "pp" in request["active"][0]["moves"][0]:
                    for move_info in request["active"][0]["moves"]:
                        move = TeamState.get_matching_move(pokemon, move_info)
                        if pokemon.maxed or just_zmoved or just_unmaxed or self.gen <= 3:
                            move.pp = move_info["pp"]
                        TeamState.check_move_consistency(move, move_info)

    @staticmethod
    def check_condition_consistency(pokemon: PokemonState, pokemon_info: Any):
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

    @staticmethod
    def check_ability_consistency(pokemon: PokemonState, pokemon_info: Any):
        if pokemon.ability != pokemon_info["baseAbility"]:
            raise RuntimeError(
                f"Mismatch of request and records. Recorded {pokemon.name} to have ability = {pokemon.ability}, but "
                f"it has ability = {pokemon_info['baseAbility']}."
            )

    @staticmethod
    def check_item_consistency(pokemon: PokemonState, pokemon_info: Any):
        if pokemon.item != (pokemon_info["item"] or None):
            raise RuntimeError(
                f"Mismatch of request and records. Recorded {pokemon.name} to have item = {pokemon.item}, but it "
                f"has item = {pokemon_info['item'] or None}."
            )

    @staticmethod
    def check_move_consistency(move: MoveState, move_info: Any):
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
    def get_matching_move(pokemon: PokemonState, move_info: Any):
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

    def process(self) -> list[float]:
        pokemon_feature_lists = [pokemon.process() for pokemon in self.__team]
        pokemon_feature_lists.extend([[0.0] * 299] * (6 - len(pokemon_feature_lists)))
        features = reduce(PokemonState.concat_features, pokemon_feature_lists)
        return features
