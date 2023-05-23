import random

import torch
import torch.nn as nn
from torch import Tensor
from websockets.exceptions import ConnectionClosedError

from env import Env
from states.state import State


class Model(nn.Module):
    def __init__(self, env: Env, alpha: float, epsilon: float, gamma: float, hidden_dims: list[int]):
        super(Model, self).__init__()  # type: ignore

        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_dim = 3606
        self.hidden_dims = hidden_dims if hidden_dims else [100]
        self.output_dim = 10

        layers: list[nn.Module] = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        for layer in self.layers:
            x = torch.relu(layer.forward(x))
        return x

    def __get_action(self, state: State) -> int | None:
        action_space = state.get_valid_action_ids()
        if action_space:
            if random.random() < self.epsilon:
                action = random.choice(action_space)
            else:
                features = torch.tensor(state.process())
                outputs = self.forward(features)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_space))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action

    def __update(self, state: State, action: int | None, reward: int, next_state: State, done: bool):
        if action:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.alpha)
            if done:
                q_target = torch.tensor(reward)
            else:
                next_features = torch.tensor(next_state.process())
                next_q_values = self.forward(next_features)
                q_target = reward + self.gamma * torch.max(next_q_values)  # type: ignore
            features = torch.tensor(state.process())
            q_values = self.forward(features)
            q_estimate = q_values[action]
            td_error = q_target - q_estimate
            loss = td_error**2
            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()

    async def run_episode(self, format_str: str) -> str | None:
        try:
            state1, state2 = await self.env.reset(format_str)
            done = False
            while not done:
                action1 = self.__get_action(state1)
                action2 = self.__get_action(state2)
                next_state1, next_state2, reward1, reward2, done = await self.env.step(
                    state1,
                    state2,
                    action1,
                    action2,
                )
                self.__update(state1, action1, reward1, next_state1, done)
                self.__update(state2, action2, reward2, next_state2, done)
                state1, state2 = next_state1, next_state2
            try:
                winner_id = state1.protocol.index("win")
            except ValueError:
                winner = None
            else:
                winner = state1.protocol[winner_id + 1].strip()
            return winner
        except ConnectionClosedError:
            self.env.logger.error("Connection closed unexpectedly")
            await self.env.setup()
            winner = None
            return winner
