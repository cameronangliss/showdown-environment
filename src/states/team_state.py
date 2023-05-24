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

    def __init__(self, ident: str, protocol: list[str], request: Any | None = None):
        self.__ident = ident
        self.gen = TeamState.get_gen(protocol)
        self.__team = (
            [
                PokemonState.from_request(pokemon_json, self.gen, self.__ident)
                for pokemon_json in request["side"]["pokemon"]
            ]
            if request
            else []
        )
        self.update(protocol, request)

    @staticmethod
    def get_gen(protocol: list[str]) -> int:
        i = protocol.index("gen")
        gen = int(protocol[i + 1].strip())
        return gen

    def get_names(self) -> list[str]:
        return [pokemon.name for pokemon in self.__team]

    def get_active(self) -> PokemonState | None:
        actives = [mon for mon in self.__team if mon.active == True]
        if len(actives) == 0:
            return None
        if len(actives) == 1:
            return actives[0]
        else:
            raise RuntimeError(f"Multiple active pokemon: {[active.name for active in actives]}")

    def update(self, protocol: list[str], request: Any | None = None):
        self.update_from_protocol(protocol)
        if request:
            self.update_from_request(request)

    def update_from_request(self, request: Any, is_init: bool = False):
        new_team_info = [
            PokemonState.from_request(pokemon_json, self.gen, self.__ident)
            for pokemon_json in request["side"]["pokemon"]
        ]
        for pokemon in self.__team:
            matching_info = [new_info for new_info in new_team_info if new_info.name == pokemon.name][0]
            # Providing new move info to a newly-transformed pokemon.
            if pokemon.transformed and not pokemon.alt_moves:
                pokemon.alt_moves = [
                    MoveState.from_request(move_json, self.gen) for move_json in request["active"][0]["moves"]
                ]
            # If active has been mistracked, we assume that the active pokemon is using an illusion.
            if pokemon.active != matching_info.active:
                pokemon.hp = matching_info.hp
                pokemon.status = matching_info.status
                pokemon.active = matching_info.active
                pokemon.illusion = True
            # Conducting harsh consistency checks if illusion pokemon isn't interfering.
            elif not pokemon.illusion:
                if pokemon.hp != matching_info.hp:
                    raise RuntimeError(
                        f"Mismatch of request and records. Recorded {pokemon.name} to have hp = {pokemon.hp}, but it has hp = {matching_info.hp}."
                    )
                elif pokemon.status != matching_info.status:
                    raise RuntimeError(
                        f"Mismatch of request and records. Recorded {pokemon.name} to have status = {pokemon.status}, but it has status = {matching_info.status}."
                    )
                # elif pokemon.ability != matching_info.ability:
                #     raise RuntimeError(
                #         f"Mismatch of request and records. Recorded {pokemon.name} to have ability = {pokemon.ability}, but it has ability = {matching_info.ability}."
                #     )
                elif pokemon.get_item() != matching_info.item:
                    raise RuntimeError(
                        f"Mismatch of request and records. Recorded {pokemon.name} to have item = {pokemon.get_item()}, but it has item = {matching_info.item}."
                    )
                elif len(pokemon.moves) > 4:
                    raise RuntimeError(
                        f"Pokemon cannot have more than 4 moves: {[move.name for move in pokemon.moves]}."
                    )
                elif pokemon.active and "active" in request and "pp" in request["active"][0]["moves"][0]:
                    for move_info in request["active"][0]["moves"]:
                        move_list = [
                            move
                            for move in pokemon.get_moves()
                            if move.identifier == MoveState.get_identifier(move_info["move"])
                        ]
                        if len(move_list) > 0:
                            move = move_list[0]
                        else:
                            raise RuntimeError(
                                f"Mismatch of request and records. Pokemon {pokemon.name} has move {MoveState.get_identifier(move_info['move'])} that isn't recorded in moveset {[move.identifier for move in pokemon.get_moves()]}."
                            )
                        if move.pp != move_info["pp"]:
                            raise RuntimeError(
                                f"Mismatch of request and records. Recorded move {move.name} of pokemon {pokemon.name} to have pp = {move.pp}, but it has pp = {move_info['pp']}."
                            )
                        elif move.disabled != move_info["disabled"]:
                            raise RuntimeError(
                                f"Mismatch of request and records. Recorded move {move.name} of pokemon {pokemon.name} to have disabled = {move.disabled}, but it has disabled = {move_info['disabled']}."
                            )

    def update_from_protocol(self, protocol: list[str]):
        protocol_lines = "|".join(protocol).split("\n")
        for line in protocol_lines:
            split_line = line.split("|")
            active_pokemon = self.get_active()
            if active_pokemon is not None and len(split_line) > 2 and split_line[2][:2] == self.__ident:
                match split_line[1]:
                    case "move":
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
                        active_pokemon.update_item(None, split_line[4:])
                    case "-transform":
                        active_pokemon.transform()
                    case "-start":
                        active_pokemon.start(split_line[3:])
                    case "-activate":
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
                    case "-ability":
                        if split_line[3] == "Pressure":
                            self.pressure = True
                    case "-item":
                        if len(split_line) > 4 and split_line[4] == "[from] move: Thief":
                            active_pokemon.update_item(None, split_line[4:])
                    case _:
                        pass

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

    def process(self) -> list[float]:
        pokemon_feature_lists = [pokemon.process() for pokemon in self.__team]
        pokemon_feature_lists.extend([[0.0] * 299] * (6 - len(pokemon_feature_lists)))
        features = reduce(PokemonState.concat_features, pokemon_feature_lists)
        return features
