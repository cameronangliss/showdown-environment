from copy import deepcopy

import torch
from actor import Actor
from critic import Critic
from memory import Memory

from showdown_environment.data.dex import movedex, typedex
from showdown_environment.showdown.base_player import BasePlayer
from showdown_environment.showdown.environment import Environment
from showdown_environment.showdown.model import Model
from showdown_environment.state.battle import Battle
from showdown_environment.state.move import Move
from showdown_environment.state.pokemon import Pokemon
from showdown_environment.state.team import Team


class Player(BasePlayer):
    username: str
    password: str
    actor: Actor
    critic: Critic
    model: Model
    memory: Memory

    def __init__(
        self,
        username: str,
        password: str,
        actor: Actor,
        critic: Critic,
        memory_length: int | None = None,
    ):
        super().__init__(username, password)
        self.actor = actor
        self.critic = critic
        self.model = Model()
        self.memory = Memory([], maxlen=memory_length)

    async def improve(self, env: Environment, num_episodes: int, min_win_rate: float):
        print("Generating experiences...")
        experiences, _ = await env.run_episodes(
            self, num_episodes, memory_length=self.memory.maxlen
        )
        self.memory.extend(experiences)
        while True:
            print(f"Training on {len(self.memory)} experiences...")
            for i in range(1000):
                batch = self.memory.sample(round(len(self.memory) / 100))
                for exp in batch:
                    self.actor.update(exp)
                print(f"Progress: {(i + 1) / 10}%", end="\r")
            print("Evaluating model...           ")
            experiences, num_wins = await env.run_episodes(
                self,
                num_episodes,
                min_win_rate=min_win_rate,
                memory_length=self.memory.maxlen,
            )
            if num_wins < round(min_win_rate * num_episodes, 5):
                print("Improvement failed.")
                self.memory.extend(experiences)
            else:
                print("Improvement succeeded!")
                self.memory.clear()
                break

    def get_action(self, state: Battle) -> int | None:
        action_space = state.get_valid_action_ids()
        mask = torch.full((10,), float("-inf")).to(self.actor.device)
        mask[action_space] = 0
        if action_space:
            inferred_state = deepcopy(state)
            inferred_state.infer_opponent_sets()
            current_score = self.critic.forward(self.encode_battle(inferred_state)).item()
            count_matrix = torch.zeros(10, 10).to(self.actor.device)
            avg_TD_matrix = torch.zeros(10, 10).to(self.actor.device)
            for _ in range(10):
                # get action
                features = self.encode_battle(inferred_state).to(self.actor.device)
                outputs = self.actor.forward(features)
                probs = torch.softmax(outputs + mask, dim=0)
                action = int(torch.multinomial(probs, num_samples=1).item())
                # get opponent's action
                opp_action = 6  # TODO: make an actual decision process for this
                # predict future with model
                new_state = self.model.predict(deepcopy(inferred_state), action, opp_action)
                # compare future state with current to see if there was improvement
                score = self.critic.forward(self.encode_battle(new_state)).item()
                td = score - current_score
                # record findings into TD matrix
                count_matrix[action][opp_action] += 1
                avg_TD_matrix[action][opp_action] += (
                    td - avg_TD_matrix[action][opp_action]
                ) / count_matrix[action][opp_action]
            avg_TD_per_action = torch.sum(count_matrix * avg_TD_matrix, dim=1) / torch.max(
                torch.sum(count_matrix, dim=1), torch.tensor(1)
            )
            final_probs = torch.softmax(avg_TD_per_action + mask, dim=0)
            action = int(torch.multinomial(final_probs, num_samples=1).item())
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
        features = [team_features, opponent_features, gen_features, weather_features]
        # print("Battle:", [len(item) for item in features])
        return torch.cat(features).to(self.actor.device)

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
        features = [*encoded_team, special_used_features]
        # print("Team:", [len(item) for item in features])
        return torch.cat(features)

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
        all_types = typedex.keys()
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
        features = [
            gender_features,
            hp_features,
            status_features,
            stats,
            type_features,
            *encoded_moves,
        ]
        # print("Pokemon:", [len(item) for item in features])
        return torch.cat(features)

    def __encode_move(self, move: Move) -> torch.Tensor:
        pp_frac_feature = move.pp / move.maxpp
        disabled_feature = float(move.is_disabled())
        details = movedex[move.identifier]
        power_feature = details["basePower"] / 250
        accuracy_feature = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
        all_types = typedex.keys()
        move_type = details["type"].lower()
        type_features = [float(t == move_type) for t in all_types]
        features = [
            pp_frac_feature,
            disabled_feature,
            power_feature,
            accuracy_feature,
            *type_features,
        ]
        return torch.tensor(features)
