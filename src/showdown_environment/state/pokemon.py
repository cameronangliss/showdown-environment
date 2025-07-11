from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from showdown_environment.data.dex import abilitydex, itemdex, movedex, pokedex, setdex
from showdown_environment.state.move import Move_


@dataclass
class Pokemon_:
    name: str
    alias: str
    identifier: str
    gen: int
    owner: str
    level: int
    gender: str | None
    types: list[str]
    hp: int
    max_hp: int
    status: str | None
    active: bool
    stats: dict[str, int]
    moves: list[Move_]
    alt_moves: list[Move_]
    ability: str | None
    item: str | None
    from_opponent: bool
    alt_types: list[str] | None = None
    alt_stats: dict[str, int] | None = None
    alt_ability: str | None = None
    item_off: bool = False
    preparing: bool = False
    tricking: bool = False
    transformed: bool = False
    illusion: bool = False
    can_mega: bool = False
    can_zmove: bool = False
    can_burst: bool = False
    can_max: bool = False
    can_tera: bool = False
    maxed: bool = False

    ###############################################################################################
    # Constructors

    @classmethod
    def from_request(cls, pokemon_json: Any, gen: int, owner: str) -> Pokemon_:
        name = pokemon_json["ident"][4:]
        alias, level, gender = Pokemon_.__parse_details(pokemon_json["details"])
        identifier = Pokemon_.get_identifier(name)
        types = [t.lower() for t in pokedex[identifier]["types"]]
        split_condition = pokemon_json["condition"].split()
        split_hp_frac = split_condition[0].split("/")
        hp = int(split_hp_frac[0])
        max_hp = int(split_hp_frac[1]) if pokemon_json["condition"] != "0 fnt" else 100
        status = split_condition[1] if len(split_condition) == 2 else None
        moves = [Move_(name, gen, "ghost" in types) for name in pokemon_json["moves"]]
        item = pokemon_json["item"] if pokemon_json["item"] != "" else None
        can_mega = (
            item is not None
            and "megaEvolves" in itemdex[item]
            and itemdex[item]["megaEvolves"] == name
        ) or (name == "Rayquaza" and "dragonascent" in pokemon_json["moves"] and gen in [6, 7])
        can_burst = name == "Necrozma" and item is not None and item == "ultranecroziumz"
        return cls(
            name=name,
            alias=alias,
            identifier=identifier,
            gen=gen,
            owner=owner,
            level=level,
            gender=gender,
            types=types,
            hp=hp,
            max_hp=max_hp,
            status=status,
            active=pokemon_json["active"],
            stats=pokemon_json["stats"],
            moves=moves,
            alt_moves=[],
            ability=pokemon_json["baseAbility"],
            item=item,
            from_opponent=False,
            can_mega=can_mega,
            can_zmove=not can_burst and item is not None and "zMove" in itemdex[item],
            can_burst=can_burst,
            can_max=gen == 8 and name not in ["Eternatus", "Zacian", "Zamazenta"],
            can_tera=gen == 9,
        )

    @classmethod
    def from_protocol(cls, name: str, details: str, gen: int, owner: str) -> Pokemon_:
        alias, level, gender = Pokemon_.__parse_details(details)
        identifier = Pokemon_.get_identifier(name)
        types = [t.lower() for t in pokedex[identifier]["types"]]
        stats = pokedex[identifier]["baseStats"]
        abilities = pokedex[identifier]["abilities"].values()
        ability_identifiers = [Pokemon_.__get_ability_identifier(ability) for ability in abilities]
        return cls(
            name=name,
            alias=alias,
            identifier=identifier,
            gen=gen,
            owner=owner,
            level=level,
            gender=gender,
            types=types,
            hp=100,
            max_hp=100,
            status=None,
            active=False,
            stats=stats,
            moves=[],
            alt_moves=[],
            ability=ability_identifiers[0] if len(ability_identifiers) == 1 else None,
            item=None,
            from_opponent=True,
        )

    ###############################################################################################
    # Parsing/String Manipulation

    def get_full_move_name(self, part_name: str) -> str:
        if part_name == "Hidden Power":
            if self.from_opponent:
                return "Hidden Power"
            else:
                return [move.name for move in self.get_moves() if move.name[:12] == "Hidden Power"][
                    0
                ]
        elif part_name[:2] == "Z-":
            return part_name[2:]
        else:
            return part_name

    @staticmethod
    def get_identifier(name: str) -> str:
        return re.sub(r"[\s\-\.\:\’]+", "", name).lower()

    @staticmethod
    def __parse_details(details: str) -> tuple[str, int, str | None]:
        # split_details format: "<alias>, <maybe level>, <maybe gender>, <maybe shiny>"
        # examples: "Castform, M, shiny", "Moltres, L84", "Raichu, L88, M"
        split_details = details.split(", ")
        alias = split_details[0]
        if alias[:-2] == "Unown":
            alias = "Unown"
        if "tera:" in split_details[-1]:
            split_details = split_details[:-1]
        if split_details[-1] == "shiny":
            split_details = split_details[:-1]
        if len(split_details) == 1:
            level = 100
            gender = None
        elif len(split_details) == 2:
            if split_details[1][0] == "L":
                level = int(split_details[1][1:])
                gender = None
            else:
                level = 100
                gender = split_details[1]
        else:
            level = int(split_details[1][1:])
            gender = split_details[2]
        return alias, level, gender

    @staticmethod
    def parse_condition(condition: str) -> tuple[int, str | None]:
        split_condition = condition.split()
        split_hp_frac = split_condition[0].split("/")
        hp = int(split_hp_frac[0])
        status = split_condition[1] if len(split_condition) == 2 else None
        return hp, status

    @staticmethod
    def __get_ability_identifier(name: str) -> str:
        return re.sub(r"[\s\-]+", "", name).lower()

    @staticmethod
    def __get_item_identifier(name: str) -> str:
        return re.sub(r"[\s\-]+", "", name).lower()

    def get_json_str(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "id": self.identifier,
                "condition": f"{self.hp}/{self.max_hp}"
                + (f" {self.status}" if self.status else ""),
                "active": self.active,
                "stats": self.get_stats(),
                "moves": [json.loads(move.get_json_str()) for move in self.get_moves()],
                "ability": self.__get_ability(),
                "item": self.item,
            }
        )

    ###############################################################################################
    # Getter methods

    def get_matching_role(self) -> Any:
        if "roles" not in setdex[self.alias]:
            return setdex[self.alias]
        else:
            roles = list(setdex[self.alias]["roles"].values())
            move_names = [move.name for move in self.get_moves()]
            matching_role_index = [
                all([self.__low_specified_move_in_list(role["moves"], move) for move in move_names])
                for role in roles
            ].index(True)
            return roles[matching_role_index]

    def __low_specified_move_in_list(self, moves: list[str], move: str) -> bool:
        if move != "Hidden Power":
            return move in moves
        else:
            return any([m[:12] == "Hidden Power" for m in moves])

    def get_types(self) -> list[str]:
        return self.types if self.alt_types is None else self.alt_types

    def get_stats(self) -> dict[str, int]:
        return self.stats if self.alt_stats is None else self.alt_stats

    def get_moves(self) -> list[Move_]:
        if self.transformed:
            return self.alt_moves
        elif "Mimic" in [move.name for move in self.moves] and len(self.alt_moves) == 1:
            return [move if move.name != "Mimic" else self.alt_moves[0] for move in self.moves]
        else:
            return self.moves

    def __get_last_used(self) -> Move_ | None:
        moves = [move for move in self.get_moves() if move.just_used]
        if len(moves) == 0:
            return None
        if len(moves) == 1:
            return moves[0]
        else:
            raise RuntimeError(
                f"Pokemon {self.name} cannot have move just used = {[move.name for move in moves]}"
            )

    def __get_ability(self) -> str | None:
        return self.alt_ability or self.ability

    def get_item(self) -> str | None:
        return None if self.item_off else self.item

    ###############################################################################################
    # Setter methods

    def __update_last_used(self, name: str):
        last_last_moves = [move for move in (self.moves + self.alt_moves) if move.just_used]
        if len(last_last_moves) == 1:
            last_last_moves[0].just_used = False
        move_last_used = [move for move in self.get_moves() if move.name == name][0]
        move_last_used.just_used = True

    def update_special_options(
        self, mega_used: bool, zmove_used: bool, burst_used: bool, max_used: bool, tera_used: bool
    ):
        if mega_used:
            self.can_mega = False
        if zmove_used:
            self.can_zmove = False
        if burst_used:
            self.can_burst = False
        if max_used:
            self.can_max = False
        if tera_used:
            self.can_tera = False

    ###############################################################################################
    # Self-updating methods used when reading through the lines of the protocol

    def switch_in(self, hp: int, status: str | None):
        if self.status == "fnt":
            raise RuntimeError(f"Cannot switch in fainted pokemon {self.name}.")
        else:
            self.hp = hp
            self.status = status
            self.active = True
            item = self.get_item()
            if item is not None:
                for move in self.get_moves():
                    move.add_item(item)

    def switch_out(self):
        current_ability = self.__get_ability()
        if (
            current_ability is not None
            and current_ability == "regenerator"
            and self.status != "fnt"
        ):
            self.hp = min(int(self.hp + self.max_hp / 3), self.max_hp)
        self.active = False
        for move in self.moves:
            move.just_used = False
            move.disable_disabled = False
            move.bide_disabled = False
            move.encore_disabled = False
            move.taunt_disabled = False
            move.remove_item()
            move.self_disabled = False
        self.alt_moves = []
        self.alt_ability = None
        self.preparing = False
        self.tricking = False
        if self.transformed:
            self.transformed = False
            self.alt_types = None
            self.alt_stats = None

    def use_move(self, name: str, info: list[str], pressure: bool):
        full_name = self.get_full_move_name(name)
        # avoiding edge cases
        if (
            full_name == "Struggle"
            or (info and info[0] == "[from]Mirror Move")
            or "isZ" in movedex[Move_.get_identifier(full_name)]
            or full_name.split()[0] in ["Max", "G-Max"]
        ):
            return
        # get move
        if full_name in [move.name for move in self.get_moves()]:
            move_used = [move for move in self.get_moves() if move.name == full_name][0]
        else:
            move_used = Move_(full_name, self.gen, "ghost" in self.get_types())
            if self.transformed:
                self.alt_moves.append(move_used)
            else:
                self.moves.append(move_used)
        # update pp
        if not (info and info[0][:16] == "[from]lockedmove"):
            pp_used = move_used.get_pp_used(pressure)
            if info and info[0] == "[from]move: Sleep Talk":
                pp_used -= 1  # accounting for sleep talk receiving two protocol messages when used
                sleep_talk = [move for move in self.get_moves() if move.name == "Sleep Talk"][0]
                sleep_talk.pp = max(sleep_talk.pp - pp_used, 0)
                return
            else:
                move_used.pp = max(move_used.pp - pp_used, 0)
        # other updates
        self.__update_last_used(move_used.name)
        for move in self.get_moves():
            move.keep_item(self.get_item())
            if move.name == "Gigaton Hammer":
                move.self_disabled = move_used.name == "Gigaton Hammer"

    def update_condition(self, hp: int, status: str | None):
        self.hp = hp
        self.status = status
        if hp == 0 and status == "fnt":
            self.alt_ability = None

    def update_ability(self, new_ability: str):
        ability = self.__get_ability()
        if ability is None:
            self.alt_ability = Pokemon_.__get_ability_identifier(new_ability)
        else:
            ability_info = abilitydex[ability]
            if not ("isPermanent" in ability_info and ability_info["isPermanent"]):
                self.alt_ability = Pokemon_.__get_ability_identifier(new_ability)

    def update_item(self, new_item: str, info: list[str]):
        if not (info and info[0] == "[from] ability: Frisk"):
            new_item_identifier = Pokemon_.__get_item_identifier(new_item)
            for move in self.get_moves():
                move.update_item(
                    self.get_item(), new_item_identifier, tricking=self.tricking, maxed=self.maxed
                )
            self.tricking = False
            self.item = new_item_identifier
            self.item_off = False

    def end_item(self, item: str, info: list[str]):
        item_identifier = Pokemon_.__get_item_identifier(item)
        if self.get_item() == item_identifier:
            for move in self.get_moves():
                move.remove_item()
            if info and info[0] == "[from] move: Knock Off" and self.gen in [3, 4]:
                self.item_off = True
            else:
                self.item = None

    def transform(self, name: str, request: Any | None):
        self.transformed = True
        self.alt_types = [t.lower() for t in pokedex[Pokemon_.get_identifier(name)]["types"]]
        if request is not None:
            new_self_info = [
                pokemon
                for pokemon in request["side"]["pokemon"]
                if pokemon["ident"][4:] == self.name
            ][0]
            self.alt_moves = [
                Move_(move_name, self.gen, "ghost" in self.get_types())
                for move_name in new_self_info["moves"]
            ]
            for move in self.alt_moves:
                move.pp = 5
            self.alt_stats = new_self_info["stats"]
            self.alt_ability = (
                new_self_info["ability"]
                if "ability" in new_self_info
                else new_self_info["baseAbility"]
            )

    def primal_reversion(self):
        if self.name == "Groudon":
            self.ability = "desolateland"
        elif self.name == "Kyogre":
            self.ability = "primordialsea"
        else:
            raise RuntimeError(f"Pokemon {self.name} cannot achieve primal reversion.")

    def mega_evolve(self, mega_stone: str | None):
        if mega_stone is None:
            mega_pokemon_identifier = "rayquazamega"
        else:
            mega_stone_identifier = Pokemon_.__get_item_identifier(mega_stone)
            mega_pokemon_identifier = Pokemon_.get_identifier(
                itemdex[mega_stone_identifier]["megaStone"]
            )
        mega_ability = pokedex[mega_pokemon_identifier]["abilities"]["0"]
        self.ability = self.__get_ability_identifier(mega_ability)
        self.alt_ability = None

    def start(self, info: list[str]):
        cause = info[0][6:] if info[0][:6] in ["move: ", "item: "] else info[0]
        match cause:
            case "Bide":
                for move in self.get_moves():
                    if move.name != "Bide":
                        move.bide_disabled = True
            case "Disable":
                move = [
                    move
                    for move in self.get_moves()
                    if move.name == self.get_full_move_name(info[1])
                ][0]
                move.disable_disabled = True
            case "Encore":
                last_used_move = self.__get_last_used()
                for move in self.get_moves():
                    if move != last_used_move:
                        move.encore_disabled = True
            case "Mimic":
                mimic_move = [move for move in self.get_moves() if move.name == "Mimic"][0]
                new_move = Move_(info[1], self.gen, "ghost" in self.get_types(), from_mimic=True)
                if self.transformed:
                    mimic_move = new_move
                else:
                    mimic_move.self_disabled = True
                    self.alt_moves.append(new_move)
            case "Taunt":
                for move in self.get_moves():
                    if movedex[move.identifier]["category"] == "Status":
                        move.taunt_disabled = True
            case "Trick":
                self.tricking = True
            case "Leppa Berry":
                move = [move for move in self.get_moves() if move.name == info[1]][0]
                move.pp = min(move.maxpp, 10)
            case "Dynamax":
                self.maxed = True
                for move in self.get_moves():
                    move.keep_item(self.get_item(), maxed=True)
            case _:
                pass

    def end(self, info: list[str]):
        cause = info[0][6:] if info[0][:6] == "move: " else info[0]
        match cause:
            case "Bide":
                for move in self.get_moves():
                    move.bide_disabled = False
            case "Disable":
                move = [move for move in self.get_moves() if move.disable_disabled][0]
                move.disable_disabled = False
            case "Encore":
                for move in self.get_moves():
                    move.encore_disabled = False
            case "Taunt":
                for move in self.get_moves():
                    move.taunt_disabled = False
            case "Dynamax":
                self.maxed = False
                for move in self.get_moves():
                    move.keep_item(self.get_item())
            case _:
                pass

    ###############################################################################################
    # Consistency checking

    def check_consistency(
        self,
        pokemon_info: Any,
        active_info: Any | None,
        zmove_pp_needs_update: bool,
        just_unmaxed: bool,
    ):
        hp, status = Pokemon_.parse_condition(pokemon_info["condition"])
        assert self.hp == hp, f"{self.hp} != {hp}"
        assert self.status == status, f"{self.status} != {status}"
        assert (
            self.get_stats() == pokemon_info["stats"]
        ), f"{self.get_stats()} != {pokemon_info['stats']}"
        assert len(self.get_moves()) <= 4, f"{len(self.get_moves())} > 4"
        if self.active and active_info is not None:
            if (
                len(self.get_moves()) == len(active_info[0]["moves"])
                and active_info[0]["moves"][0]["move"] != "Struggle"
            ):
                for move, move_info in zip(self.get_moves(), active_info[0]["moves"]):
                    move.check_consistency(
                        move_info, zmove_pp_needs_update, self.maxed, just_unmaxed
                    )
            assert self.can_mega == (
                "canMegaEvo" in active_info[0]
            ), f"{self.can_mega} != {'canMegaEvo' in active_info[0]}"
            assert self.can_zmove == (
                "canZMove" in active_info[0]
            ), f"{self.can_zmove} != {'canZMove' in active_info[0]}"
            assert self.can_burst == (
                "canUltraBurst" in active_info[0]
            ), f"{self.can_burst} != {'canUltraBurst' in active_info[0]}"
            assert self.can_max == (
                "canDynamax" in active_info[0]
            ), f"{self.can_max} != {'canDynamax' in active_info[0]}"
            assert self.can_tera == (
                "canTerastallize" in active_info[0]
            ), f"{self.can_tera} != {'canTerastallize' in active_info[0]}"
        assert (
            self.ability == pokemon_info["baseAbility"]
        ), f"{self.ability} != {pokemon_info['baseAbility']}"
        if "ability" in pokemon_info:
            assert (
                self.__get_ability() == pokemon_info["ability"]
            ), f"{self.__get_ability} != {pokemon_info['ability']}"
        assert self.item == (
            pokemon_info["item"] or None
        ), f"{self.item} != {pokemon_info['item'] or None}"
