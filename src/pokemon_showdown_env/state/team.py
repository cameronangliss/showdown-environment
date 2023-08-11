import json
from functools import reduce
from typing import Any

from pokemon_showdown_env.state.pokemon import Pokemon


class Team:
    __ident: str
    __gen: int
    __team: list[Pokemon]
    __pressure: bool = False
    __mega_used: bool = False
    __zmove_used: bool = False
    __zmove_pp_updated: bool = False
    __burst_used: bool = False
    __max_used: bool = False
    __tera_used: bool = False

    def __init__(self, ident: str, gen: int, protocol: list[str], request: Any | None = None):
        self.__ident = ident
        self.__gen = gen
        self.__team = (
            [
                Pokemon.from_request(pokemon_json, self.__gen, self.__ident)
                for pokemon_json in request["side"]["pokemon"]
            ]
            if request
            else []
        )
        self.update(protocol, request)

    ###################################################################################################################
    # Getter methods

    def get_active(self) -> Pokemon | None:
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
        protocol_lines = "|".join(protocol).split("\n")
        for line in protocol_lines:
            split_line = line.split("|")
            active_pokemon = self.get_active()
            if active_pokemon is None:
                if len(split_line) > 2 and split_line[1] == "switch" and split_line[2][:2] == self.__ident:
                    hp, status = Pokemon.parse_condition(split_line[4])
                    self.__switch(split_line[2][5:], split_line[3], hp, status)
            elif len(split_line) > 2 and split_line[2][:2] == self.__ident:
                self.__update_with_player_message(split_line, active_pokemon, request)
            elif len(split_line) > 2 and split_line[2][:2] != self.__ident:
                self.__update_with_opponent_message(split_line, active_pokemon)
            if active_pokemon is not None:
                active_pokemon.update_special_options(
                    self.__mega_used, self.__zmove_used, self.__burst_used, self.__max_used, self.__tera_used
                )
        active_pokemon = self.get_active()
        if active_pokemon is not None:
            active_pokemon.tricking = False
        if request:
            active_pokemon = self.get_active()
            active_name = active_pokemon.name if active_pokemon else ""
            just_unmaxed = f"|-end|{self.__ident}a: {active_name}|Dynamax" in protocol_lines
            team_info = [pokemon_info for pokemon_info in request["side"]["pokemon"]]
            for pokemon, pokemon_info in zip(self.__team, team_info):
                # If active has been mistracked, we assume that the active pokemon is using an illusion.
                if pokemon.active != pokemon_info["active"]:
                    hp, status = Pokemon.parse_condition(pokemon_info["condition"])
                    pokemon.hp = hp
                    pokemon.status = status
                    pokemon.active = pokemon_info["active"]
                    pokemon.illusion = True
            if not any([pokemon.illusion for pokemon in self.__team]):
                self.check_consistency(request, just_unmaxed)

    def __update_with_player_message(self, split_line: list[str], active_pokemon: Pokemon, request: Any | None):
        match split_line[1]:
            case "move":
                if not active_pokemon.preparing:
                    active_pokemon.use_move(split_line[3], split_line[5:], self.__pressure)
                active_pokemon.preparing = False
            case "switch":
                hp, status = Pokemon.parse_condition(split_line[4])
                self.__switch(split_line[2][5:], split_line[3], hp, status)
            case "drag":
                hp, status = Pokemon.parse_condition(split_line[4])
                self.__switch(split_line[2][5:], split_line[3], hp, status)
            case "faint":
                fainted_pokemon_name = split_line[2][split_line[2].index(" ") + 1 :]
                fainted_pokemon = [pokemon for pokemon in self.__team if pokemon.name == fainted_pokemon_name][0]
                fainted_pokemon.update_condition(0, "fnt")
            case "replace":
                self.__replace(split_line[2][5:], split_line[3])
            case "-damage":
                hp, status = Pokemon.parse_condition(split_line[3])
                active_pokemon.update_condition(hp, status)
            case "-heal":
                healed_pokemon_name = split_line[2][split_line[2].index(" ") + 1 :]
                healed_pokemon = [pokemon for pokemon in self.__team if pokemon.name == healed_pokemon_name][0]
                hp, status = Pokemon.parse_condition(split_line[3])
                healed_pokemon.update_condition(hp, status)
            case "-sethp":
                hp, status = Pokemon.parse_condition(split_line[3])
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
            case "-formechange":
                if request is not None:
                    new_pokemon_info = [pokemon for pokemon in request["side"]["pokemon"] if pokemon["active"]][0]
                    active_pokemon.alt_stats = new_pokemon_info["stats"]
            case "-ability":
                active_pokemon.update_ability(split_line[3])
            case "-item":
                active_pokemon.update_item(split_line[3], split_line[4:])
            case "-enditem":
                active_pokemon.end_item(split_line[3], split_line[4:])
            case "-prepare":
                active_pokemon.preparing = True
            case "-anim":
                active_pokemon.preparing = False
            case "cant":
                active_pokemon.preparing = False
            case "-mega":
                self.__mega_used = True
                active_pokemon.mega_evolve(split_line[4] or None)
            case "-primal":
                active_pokemon.primal_reversion()
            case "-zpower":
                self.__zmove_used = True
            case "-burst":
                self.__burst_used = True
            case "-terastallize":
                self.__tera_used = True
            case "-transform":
                copied_pokemon_name = split_line[3][split_line[3].index(" ") + 1 :]
                active_pokemon.transform(copied_pokemon_name, request)
            case "-start":
                if len(split_line) > 3 and split_line[3] == "Dynamax":
                    self.__max_used = True
                active_pokemon.start(split_line[3:])
            case "-activate":
                if len(split_line) > 3:
                    if split_line[3] == "ability: Mummy" and self.__pressure:
                        self.__pressure = False
                    if split_line[3] == "Dynamax":
                        self.__max_used = True
                active_pokemon.start(split_line[3:])
            case "-end":
                active_pokemon.end(split_line[3:])
            case _:
                pass

    def __update_with_opponent_message(self, split_line: list[str], active_pokemon: Pokemon):
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

    def __switch(self, incoming_pokemon_name: str, details: str, hp: int, status: str | None):
        # get incoming pokemon
        if incoming_pokemon_name in [pokemon.name for pokemon in self.__team]:
            incoming_pokemon = [pokemon for pokemon in self.__team if pokemon.name == incoming_pokemon_name][0]
            incoming_index = self.__team.index(incoming_pokemon)
        else:
            incoming_pokemon = Pokemon.from_protocol(incoming_pokemon_name, details, self.__gen, self.__ident)
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
                actual_pokemon = Pokemon.from_protocol(pokemon, details, self.__gen, self.__ident)
                actual_pokemon.switch_in(active_pokemon.hp, active_pokemon.status)
                active_pokemon = actual_pokemon
        else:
            raise RuntimeError("Cannot replace pokemon if there are no active pokemon.")

    ###################################################################################################################
    # Consistency checking

    def check_consistency(self, request: Any, just_unmaxed: bool):
        team_info = [pokemon_info for pokemon_info in request["side"]["pokemon"]]
        for pokemon, pokemon_info in zip(self.__team, team_info):
            zmove_pp_needs_update = self.__zmove_used and not self.__zmove_pp_updated
            active_info = request["active"] if "active" in request else None
            pokemon.check_consistency(pokemon_info, active_info, zmove_pp_needs_update, just_unmaxed)
            if zmove_pp_needs_update:
                self.__zmove_pp_updated = True

    ###################################################################################################################
    # Processes TeamState object into a feature vector to be fed into the model's input layer

    def process(self) -> list[float]:
        pokemon_feature_lists = [pokemon.process() for pokemon in self.__team]
        pokemon_feature_lists.extend([[0.0] * 123] * (6 - len(pokemon_feature_lists)))
        pokemon_features = reduce(lambda features1, features2: features1 + features2, pokemon_feature_lists)
        special_used_features = [
            float(attribute)
            for attribute in [
                self.__mega_used,
                self.__zmove_used,
                self.__max_used,
                self.__tera_used,
            ]
        ]
        features = pokemon_features + special_used_features
        return features
