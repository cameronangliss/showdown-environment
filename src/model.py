from __future__ import annotations

import random
from copy import deepcopy
from datetime import datetime
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from websockets.exceptions import ConnectionClosedError

from env import Env
from state.state import State


class Experience(NamedTuple):
    state: State
    action: int | None
    next_state: State
    reward: int
    done: bool


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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.__alpha)

    def __forward(self, x: Tensor) -> Tensor:  # type: ignore
        for layer in self.__layers:
            x = torch.relu(layer.forward(x))
        return x

    def __update(self, experience: Experience):
        if experience.action is not None:
            if experience.done:
                q_target = torch.tensor(experience.reward)
            else:
                next_features = torch.tensor(experience.next_state.process())
                next_q_values = self.__forward(next_features)
                q_target = experience.reward + self.__gamma * torch.max(next_q_values)  # type: ignore
            features = torch.tensor(experience.state.process())
            q_values = self.__forward(features)
            q_estimate = q_values[experience.action]
            td_error = q_target - q_estimate
            loss = td_error**2
            self.optimizer.zero_grad()
            loss.backward()  # type: ignore
            self.optimizer.step()

    ###################################################################################################################
    # Training methods

    async def improve(self):
        experiences = []
        num_wins = 0
        while num_wins < 55:
            experiences, num_wins = await self.attempt_improve(experiences)

    async def attempt_improve(self, experiences: list[Experience]) -> tuple[list[Experience], int]:
        duplicate_model = deepcopy(self)
        # gathering data
        print("Gathering experiences...")
        new_experiences, _ = await self.__run_episodes(duplicate_model)
        experiences += new_experiences
        # training
        print(f"Training on {len(experiences)} experiences...")
        for _ in range(1000):
            experience_sample = random.sample(experiences, k=round(len(experiences) / 100))
            for experience in experience_sample:
                self.__update(experience)
        # evaluating
        _, num_wins = await self.__run_episodes(duplicate_model)
        print(f"Win rate: {num_wins}/100")
        if num_wins < 55:
            print("Improvement failed.")
            self.__dict__ = duplicate_model.__dict__
        else:
            print("Improvement succeeded!")
        return experiences, num_wins

    async def __run_episodes(self, alt_model: Model) -> tuple[list[Experience], int]:
        # formats = [f"gen{i}randombattle" for i in range(1, 5)]
        env = Env()
        await env.setup()
        experiences: list[Experience] = []
        num_wins = 0
        for i in range(100):
            new_experiences, winner = await self.__run_episode(alt_model, env, "gen4randombattle")
            experiences += new_experiences
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: {winner} wins game {i + 1}")
            if winner == env.player.username:
                num_wins += 1
        await env.close()
        meaningful_experiences = list(filter(lambda experience: experience.action is not None, experiences))
        return meaningful_experiences, num_wins

    async def __run_episode(self, alt_model: Model, env: Env, format_str: str) -> tuple[list[Experience], str | None]:
        experiences: list[Experience] = []
        try:
            state1, state2 = await env.reset(format_str)
            done = False
            while not done:
                action1 = self.__get_action(state1)
                action2 = alt_model.__get_action(state2)
                next_state1, next_state2, reward1, reward2, done = await env.step(
                    state1,
                    state2,
                    action1,
                    action2,
                )
                experience1 = Experience(state1, action1, next_state1, reward1, done)
                experience2 = Experience(state2, action2, next_state2, reward2, done)
                experiences += [experience1, experience2]
                state1, state2 = next_state1, next_state2
            try:
                winner_id = state1.protocol.index("win")
            except ValueError:
                winner = None
            else:
                winner = state1.protocol[winner_id + 1].strip()
            return experiences, winner
        except ConnectionClosedError:
            env.logger.error("Connection closed unexpectedly")
            await env.setup()
            winner = None
            return [], winner

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
