from __future__ import annotations

import random
from datetime import datetime

import torch
import torch.nn as nn
from torch import Tensor
from websockets.exceptions import ConnectionClosedError

from env import Env
from states.state import State


class Model(nn.Module):
    __alpha: float
    __epsilon: float
    __gamma: float
    __input_dim: int
    __hidden_dims: list[int]
    __output_dim: int
    __layers: nn.ModuleList

    def __init__(self, alpha: float, epsilon: float, gamma: float, hidden_dims: list[int]):
        super(Model, self).__init__()  # type: ignore

        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__gamma = gamma
        self.__input_dim = 1502
        self.__hidden_dims = hidden_dims if hidden_dims else [100]
        self.__output_dim = 26

        layers: list[nn.Module] = []
        layers.append(nn.Linear(self.__input_dim, self.__hidden_dims[0]))
        for i in range(len(self.__hidden_dims) - 1):
            layers.append(nn.Linear(self.__hidden_dims[i], self.__hidden_dims[i + 1]))
        layers.append(nn.Linear(self.__hidden_dims[-1], self.__output_dim))
        self.__layers = nn.ModuleList(layers)

    def __forward(self, x: Tensor) -> Tensor:  # type: ignore
        for layer in self.__layers:
            x = torch.relu(layer.forward(x))
        return x

    def __update(self, state: State, action: int | None, reward: int, next_state: State, done: bool):
        if action:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.__alpha)
            if done:
                q_target = torch.tensor(reward)
            else:
                next_features = torch.tensor(next_state.process())
                next_q_values = self.__forward(next_features)
                q_target = reward + self.__gamma * torch.max(next_q_values)  # type: ignore
            features = torch.tensor(state.process())
            q_values = self.__forward(features)
            q_estimate = q_values[action]
            td_error = q_target - q_estimate
            loss = td_error**2
            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()

    ###################################################################################################################
    # Training methods

    async def attempt_improve(self, alt_model: Model, env: Env, train_episodes: int, eval_episodes: int) -> float:
        # training against frozen state of self
        for i in range(train_episodes):
            winner = await self.__run_episode(alt_model, env, "gen4randombattle")
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: {winner} wins game {i + 1}")
        # evaluating against frozen state of self
        print(f"Trained for {train_episodes} games. Now evaluating progress...")
        num_wins = 0
        for i in range(eval_episodes):
            winner = await self.__run_episode(alt_model, env, "gen4randombattle", frozen=True)
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: {winner} wins game {i + 1}")
            if winner == env.player.username:
                num_wins += 1
        # returning win rate
        print(f"Win rate: {num_wins}/{eval_episodes}")
        return num_wins / eval_episodes

    async def __run_episode(self, alt_model: Model, env: Env, format_str: str, frozen: bool = False) -> str | None:
        try:
            state1, state2 = await env.reset(format_str)
            done = False
            while not done:
                action1 = self.__get_action(state1)
                action2 = alt_model.__get_action(state2)
                next_state1, next_state2, reward, done = await env.step(
                    state1,
                    state2,
                    action1,
                    action2,
                )
                if not frozen:
                    self.__update(state1, action1, reward, next_state1, done)
                state1, state2 = next_state1, next_state2
            try:
                winner_id = state1.protocol.index("win")
            except ValueError:
                winner = None
            else:
                winner = state1.protocol[winner_id + 1].strip()
            return winner
        except ConnectionClosedError:
            env.logger.error("Connection closed unexpectedly")
            await env.setup()
            winner = None
            return winner

    def __get_action(self, state: State) -> int | None:
        action_space = state.get_valid_action_ids()
        if action_space:
            if random.random() < self.__epsilon:
                action = random.choice(action_space)
            else:
                features = torch.tensor(state.process())
                outputs = self.__forward(features)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_space))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action
