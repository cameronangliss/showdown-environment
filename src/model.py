import random

import torch
import torch.nn as nn
from torch import Tensor

from observation import Observation


class Model(nn.Module):
    def __init__(self, alpha: float, epsilon: float, gamma: float, hidden_dims: list[int]):
        super(Model, self).__init__()  # type: ignore

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_dim = 716
        self.hidden_dims = hidden_dims if hidden_dims else [100]
        self.output_dim = 10

        layers: list[nn.Module] = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, obs: Observation) -> Tensor:  # type: ignore
        x = torch.tensor(obs.process())
        for layer in self.layers:
            x = torch.relu(layer.forward(x))
        return x

    def get_action(self, obs: Observation) -> int | None:
        action_space = obs.get_valid_action_ids()
        if action_space:
            if random.random() < self.epsilon:
                action = random.choice(action_space)
            else:
                outputs = self.forward(obs)
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
                next_q_values = self.forward(next_obs)
                q_target = reward + self.gamma * torch.max(next_q_values)  # type: ignore
            q_values = self.forward(obs)
            q_estimate = q_values[action]
            td_error = q_target - q_estimate
            loss = td_error**2
            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()
