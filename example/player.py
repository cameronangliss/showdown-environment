import random

import torch
from model import Model

from showdown_environment.data.dex import movedex, typedex
from showdown_environment.showdown.base_player import BasePlayer
from showdown_environment.showdown.environment import Environment
from showdown_environment.state.battle import Battle
from showdown_environment.state.move import Move
from showdown_environment.state.pokemon import Pokemon
from showdown_environment.state.team import Team


class Player(BasePlayer):
    username: str
    password: str
    model: Model

    def __init__(self, username: str, password: str, model: Model):
        super().__init__(username, password)
        self.model = model

    async def improve(self, env: Environment, num_episodes: int, min_win_rate: float):
        print("Generating experiences...")
        experiences, _ = await env.run_episodes(
            self, num_episodes, memory_length=self.model.memory_length
        )
        self.model.memory.extend(experiences)
        while True:
            print(f"Training on {len(self.model.memory)} experiences...")
            for i in range(1000):
                batch = self.model.memory.sample(round(len(self.model.memory) / 100))
                for exp in batch:
                    self.model.update(exp)
                print(f"Progress: {(i + 1) / 10}%", end="\r")
            print("Evaluating model...           ")
            experiences, num_wins = await env.run_episodes(
                self,
                num_episodes,
                min_win_rate=min_win_rate,
                memory_length=self.model.memory_length,
            )
            if num_wins < min_win_rate * num_episodes:
                print("Improvement failed.")
                self.model.memory.extend(experiences)
            else:
                print("Improvement succeeded!")
                self.model.memory.clear()
                break

    def get_action(self, state: Battle) -> int | None:
        action_space = state.get_valid_action_ids()
        if action_space:
            if random.random() < self.model.epsilon:
                action = random.choice(action_space)
            else:
                features = self.encode_battle(state).to(self.model.device)
                outputs = self.model.forward(features)
                valid_outputs = torch.index_select(
                    outputs, dim=0, index=torch.tensor(action_space).to(self.model.device)
                )
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action

    def encode_battle(self, battle: Battle) -> torch.Tensor:
        team_features = self.__encode_team(battle.team)
        opponent_features = self.__encode_team(battle.opponent_team)
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
        weather_features = torch.tensor(
            [float(weather == weather_type) for weather_type in weather_types]
        )
        features = torch.cat([team_features, opponent_features, gen_features, weather_features])
        return features.to(self.model.device)

    def __encode_team(self, team: Team) -> torch.Tensor:
        encoded_team = [self.__encode_pokemon(pokemon) for pokemon in team.team]
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

    def __encode_pokemon(self, pokemon: Pokemon) -> torch.Tensor:
        gender_features = torch.tensor(
            [
                float(gender_bool)
                for gender_bool in [
                    pokemon.gender == "M",
                    pokemon.gender == "F",
                    pokemon.gender == None,
                ]
            ]
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
            [
                stat / 255 if pokemon.from_opponent else stat / 1000
                for stat in pokemon.get_stats().values()
            ]
        )
        encoded_moves = [self.__encode_move(move) for move in pokemon.get_moves()]
        encoded_moves += [torch.zeros(22)] * (4 - len(encoded_moves))
        return torch.cat(
            [gender_features, hp_features, status_features, stats, type_features, *encoded_moves]
        )

    def __encode_move(self, move: Move) -> torch.Tensor:
        pp_frac_feature = move.pp / move.maxpp
        disabled_feature = float(move.is_disabled())
        details = movedex[f"gen{move.gen}"][move.identifier]
        power_feature = details["basePower"] / 250
        accuracy_feature = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
        all_types = typedex[f"gen{move.gen}"].keys()
        move_type = details["type"].lower()
        type_features = [float(t == move_type) for t in all_types]
        return torch.tensor(
            [pp_frac_feature, disabled_feature, power_feature, accuracy_feature, *type_features]
        )
