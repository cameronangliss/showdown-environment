import random
import torch
from torch import Tensor
import torch.nn as nn

from player import Observation


class Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def process_observation(self, obs: Observation) -> torch.Tensor:
        return torch.rand(self.input_dim)

    def forward(self, obs: Observation) -> torch.Tensor:
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

    def get_action(self, obs: Observation, epsilon: float) -> int | None:
        action_space = Model.get_valid_action_ids(obs)
        if action_space:
            if random.random() < epsilon:
                action = random.choice(action_space)
            else:
                outputs = self(obs)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_space))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action

    def update(
        self,
        obs: Observation,
        action: int | None,
        reward: int,
        next_obs: Observation,
        done: bool,
        gamma: float,
        alpha: float,
    ):
        if action:
            optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
            if done:
                q_target = torch.tensor(reward)
            else:
                next_q_values: Tensor = self(next_obs)
                q_target = reward + gamma * torch.max(next_q_values)
            q_values: Tensor = self(obs)
            q_estimate = q_values[action]
            td_error = q_target - q_estimate
            loss = td_error**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
