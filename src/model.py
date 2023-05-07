import json
import random
import re
import torch
from torch import Tensor
import torch.nn as nn
from typing import Any

from player import Observation


with open("json/pokedex.json") as f:
    pokedex = json.load(f)


with open("json/moves.json") as f:
    movedex = json.load(f)


with open("json/typechart.json") as f:
    typechart = json.load(f)


def frac_str_to_float(string: str) -> float:
    if string == "0 fnt":
        return 0
    else:
        frac_str = re.sub(r"[A-Za-z\s]+", "", string)
        numerator, denominator = map(float, frac_str.split("/"))
        return numerator / denominator


class Model(nn.Module):
    def __init__(self, epsilon: float, gamma: float, alpha: float, hidden_dim: int) -> None:
        super(Model, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.input_dim = 620
        self.hidden_dim = hidden_dim
        self.output_dim = 10
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)

    def process_observation(self, obs: Observation) -> Tensor:
        if "wait" in obs.request or "forceSwitch" in obs.request:
            return torch.rand(self.input_dim)
        else:
            active_moves = obs.request["active"][0]["moves"]
            team_features = sum(
                [Model.process_pokemon(active_moves, pokemon) for pokemon in obs.request["side"]["pokemon"]], []
            )
            types = typechart.keys()
            opponent_id = "p2" if obs.request["side"]["id"] == "p1" else "p1"
            opponent_info = [
                re.sub(r"[\-\.\’\s]+", "", msg[5:]).lower() for msg in obs.protocol if msg[:3] == f"{opponent_id}a"
            ]
            if opponent_info:
                opponent = opponent_info[0]
                opponent_details = pokedex[opponent]
                opponent_base_stats = [stat / 255 for stat in opponent_details["baseStats"].values()]
                opponent_types = [(1.0 if t in opponent_details["types"] else 0.0) for t in types]
            else:
                opponent_base_stats = [0.0] * 6
                opponent_types = [0.0] * 18
            return torch.tensor(team_features + opponent_base_stats + opponent_types)

    @staticmethod
    def process_pokemon(active_moves: Any, pokemon: Any) -> list[float]:
        if not pokemon:
            return [0.0] * 98
        else:
            name = re.sub(r"[\-\.\’\s]+", "", pokemon["ident"][4:]).lower()
            details = pokedex[name]
            types = typechart.keys()
            pokemon_types = [float(t in details["types"]) for t in types]
            moves = pokemon["moves"]
            moves.extend([None] * (4 - len(moves)))
            move_features = sum([Model.process_move(move) for move in moves], [])
            if pokemon["active"] == True:
                active_move_ids = [move["id"] for move in active_moves]
                active_features = []
                for i in range(4):
                    if i < len(active_move_ids) and active_move_ids[i] in moves:
                        active_features += [active_moves[i]["pp"] / active_moves[i]["maxpp"]]
                        active_features += [float(active_moves[i]["disabled"])]
                    else:
                        active_features += [0.0, 0.0]
                features = pokemon_types + move_features + active_features
            else:
                features = pokemon_types + move_features
            return features

    @staticmethod
    def process_move(move: str) -> list[float]:
        if not move:
            return [0.0] * 20
        else:
            formatted_move = re.sub(r"\d+$", "", move)
            details = movedex[formatted_move]
            power = details["basePower"] / 250
            accuracy = 1.0 if details["accuracy"] == True else details["accuracy"] / 100
            types = typechart.keys()
            move_types = [float(t in details["type"]) for t in types]
            return [power, accuracy] + move_types

    def forward(self, obs: Observation) -> Tensor:
        x = self.process_observation(obs)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def get_valid_action_ids(obs: Observation) -> list[int]:
        valid_switch_ids = [
            i + 4
            for i, pokemon in enumerate(obs.request["side"]["pokemon"])
            if not pokemon["active"] and pokemon["condition"] != "0 fnt"
        ]
        if "wait" in obs.request:
            valid_action_ids = []
        elif "forceSwitch" in obs.request:
            if "Revival Blessing" in obs.protocol:
                dead_switch_ids = [switches for switches in range(4, 10) if switches not in valid_switch_ids]
                valid_action_ids = dead_switch_ids
            else:
                valid_action_ids = valid_switch_ids
        else:
            valid_move_ids = [
                i
                for i, move in enumerate(obs.request["active"][0]["moves"])
                if not ("disabled" in move and move["disabled"])
            ]
            if "trapped" in obs.request["active"][0] or "maybeTrapped" in obs.request["active"][0]:
                valid_action_ids = valid_move_ids
            else:
                valid_action_ids = valid_move_ids + valid_switch_ids
        return valid_action_ids

    def get_action(self, obs: Observation) -> int | None:
        action_space = Model.get_valid_action_ids(obs)
        if action_space:
            if random.random() < self.epsilon:
                action = random.choice(action_space)
            else:
                outputs = self(obs)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_space))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action

    def update(self, obs: Observation, action: int | None, reward: int, next_obs: Observation, done: bool):
        if action:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.alpha)
            if done:
                q_target = torch.tensor(reward)
            else:
                next_q_values: Tensor = self(next_obs)
                q_target = reward + self.gamma * torch.max(next_q_values)
            q_values: Tensor = self(obs)
            q_estimate = q_values[action]
            td_error = q_target - q_estimate
            loss = td_error**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
