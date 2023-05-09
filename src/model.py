import random
import torch
from torch import Tensor
import torch.nn as nn

from observation_parser import ObservationParser
from player import Observation


class Model(nn.Module):
    def __init__(self, alpha: float, epsilon: float, gamma: float, *hidden_dims: int):
        super(Model, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_dim = 716
        self.hidden_dims = hidden_dims
        self.output_dim = 10
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        self.layers = nn.ModuleList(layers)
        self.parser = ObservationParser()

    def forward(self, obs: Observation) -> Tensor:
        x = torch.tensor(self.parser.process_observation(obs))
        for layer in self.layers:
            x = torch.relu(layer(x))
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
