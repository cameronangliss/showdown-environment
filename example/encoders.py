import sys

import torch

sys.path.append(".")

from showdown_environment.data.dex import movedex, typedex
from showdown_environment.state.battle import Battle
from showdown_environment.state.move import Move
from showdown_environment.state.pokemon import Pokemon
from showdown_environment.state.team import Team


def encode_battle(battle: Battle) -> torch.Tensor:
    team_features = __encode_team(battle.team)
    opponent_features = __encode_team(battle.opponent_team)
    gen_features = torch.tensor([float(n == battle.gen) for n in range(1, 10)])
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
    if "-weather" in battle.protocol:
        weather = battle.protocol[battle.protocol.index("-weather") + 1]
    else:
        weather = None
    weather_features = torch.tensor([float(weather == weather_type) for weather_type in weather_types])
    return torch.cat([team_features, opponent_features, gen_features, weather_features])


def __encode_team(team: Team) -> torch.Tensor:
    encoded_team = [__encode_pokemon(pokemon) for pokemon in team.team]
    encoded_team += [torch.zeros(123)] * (6 - len(encoded_team))
    special_used_features = torch.tensor(
        [
            float(attribute)
            for attribute in [
                team.mega_used,
                team.zmove_used,
                team.burst_used,
                team.max_used,
                team.tera_used,
            ]
        ]
    )
    return torch.cat([*encoded_team, special_used_features])


def __encode_pokemon(pokemon: Pokemon) -> torch.Tensor:
    gender_features = torch.tensor(
        [float(gender_bool) for gender_bool in [pokemon.gender == "M", pokemon.gender == "F", pokemon.gender == None]]
    )
    all_types = typedex[f"gen{pokemon.gen}"].keys()
    type_features = torch.tensor([float(t in pokemon.get_types()) for t in all_types])
    hp_features = torch.tensor(
        (
            [pokemon.hp / pokemon.max_hp]
            if pokemon.from_opponent
            else [pokemon.hp / pokemon.max_hp, pokemon.max_hp / 1000]
        )
    )
    status_conditions = ["psn", "tox", "par", "slp", "brn", "frz", "fnt"]
    status_features = torch.tensor(
        [float(pokemon.status == status_condition) for status_condition in status_conditions]
    )
    stats = torch.tensor(
        [stat / 255 if pokemon.from_opponent else stat / 1000 for stat in pokemon.get_stats().values()]
    )
    encoded_moves = [__encode_move(move) for move in pokemon.get_moves()]
    encoded_moves += [torch.zeros(22)] * (4 - len(encoded_moves))
    return torch.cat([gender_features, hp_features, status_features, stats, type_features, *encoded_moves])


def __encode_move(move: Move) -> torch.Tensor:
    pp_frac_feature = move.pp / move.maxpp
    disabled_feature = float(move.is_disabled())
    details = movedex[f"gen{move.gen}"][move.identifier]
    power_feature = details["basePower"] / 250
    accuracy_feature = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
    all_types = typedex[f"gen{move.gen}"].keys()
    move_type = details["type"].lower()
    type_features = [float(t == move_type) for t in all_types]
    return torch.tensor([pp_frac_feature, disabled_feature, power_feature, accuracy_feature, *type_features])
