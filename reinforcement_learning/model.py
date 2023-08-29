from __future__ import annotations

import random
from copy import deepcopy
from datetime import datetime
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from poke_dojo.showdown.environment import Environment
from poke_dojo.state.battle import Battle


class Experience(NamedTuple):
    turn: int
    total_turns: int
    state_tensor: Tensor
    action: int | None
    next_state_tensor: Tensor
    reward: int
    done: bool


class Model(nn.Module):
    __alpha: float
    __epsilon: float
    __gamma: float
    __layers: nn.ModuleList

    def __init__(self, alpha: float, epsilon: float, gamma: float):
        super().__init__()  # type: ignore
        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__gamma = gamma
        self.layers = nn.Sequential(
            nn.Linear(1502, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 26)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.__alpha)

        # Move the model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def __forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.layers(x)

    def __update(self, experience: Experience):
        if experience.action is not None:
            if experience.done:
                q_target = torch.tensor(experience.reward)
            else:
                next_q_values = self.__forward(experience.next_state_tensor)
                q_target = experience.reward + self.__gamma * torch.max(next_q_values)  # type: ignore
            q_values = self.__forward(experience.state_tensor)
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
        new_experiences, _ = await self.__run_episodes(duplicate_model, 100)
        experiences += new_experiences
        # training
        choose_weights = [1 if exp.turn == exp.total_turns else 1 / exp.total_turns for exp in experiences]
        print(f"Training on {len(experiences)} experiences.")
        print(f"Progress: 0.0%", end="\r")
        for i in range(1000):
            experience_sample = random.choices(experiences, weights=choose_weights, k=round(len(experiences) / 100))
            for experience in experience_sample:
                self.__update(experience)
            print(f"Progress: {(i + 1) / 10}%", end="\r")
        # evaluating
        _, num_wins = await self.__run_episodes(duplicate_model, 100, min_win_rate=0.55)
        print(f"Win rate: {num_wins}/100")
        if num_wins < 55:
            print("Improvement failed.")
            self.__dict__ = duplicate_model.__dict__
        else:
            print("Improvement succeeded!")
        return experiences, num_wins

    async def __run_episodes(
        self, alt_model: Model, num_episodes: int, min_win_rate: float | None = None
    ) -> tuple[list[Experience], int]:
        # formats = [f"gen{i}randombattle" for i in range(1, 5)]
        env = Environment()
        await env.setup()
        experiences: list[Experience] = []
        num_wins = 0
        for i in range(num_episodes):
            new_experiences, winner = await self.__run_episode(alt_model, env, "gen4randombattle")
            experiences += new_experiences
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: {winner} wins game {i + 1}")
            if winner == env.agent.username:
                num_wins += 1
            if min_win_rate is not None and (
                num_wins / num_episodes >= min_win_rate or (i - num_wins) / num_episodes > 1 - min_win_rate
            ):
                break
        await env.close()
        meaningful_experiences = list(filter(lambda experience: experience.action is not None, experiences))
        return meaningful_experiences, num_wins

    async def __run_episode(
        self, alt_model: Model, env: Environment, format_str: str
    ) -> tuple[list[Experience], str | None]:
        experiences: list[Experience] = []
        try:
            state1, state2 = await env.reset(format_str)
            turn = 0
            done = False
            while not done:
                turn += 1
                action1 = self.__get_action(state1)
                action2 = alt_model.__get_action(state2)
                next_state1, next_state2, reward1, reward2, done = await env.step(
                    state1,
                    state2,
                    action1,
                    action2,
                )
                experience1 = Experience(
                    turn,
                    0,  # temporary value
                    torch.tensor(state1.process()).to(self.device),
                    action1,
                    torch.tensor(next_state1.process()).to(self.device),
                    reward1,
                    done,
                )
                experience2 = Experience(
                    turn,
                    0,  # temporary value
                    torch.tensor(state2.process()).to(self.device),
                    action2,
                    torch.tensor(next_state2.process()).to(self.device),
                    reward2,
                    done,
                )
                experiences += [experience1, experience2]
                state1, state2 = next_state1, next_state2
            experiences = [exp._replace(total_turns=turn) for exp in experiences]
            try:
                winner_id = state1.protocol.index("win")
            except ValueError:
                winner = None
            else:
                winner = state1.protocol[winner_id + 1].strip()
            return experiences, winner
        except (ConnectionClosedError, ConnectionClosedOK):
            env.logger.error("Connection closed unexpectedly")
            await env.setup()
            winner = None
            return [], winner

    def __get_action(self, state: Battle) -> int | None:
        action_space = state.get_valid_action_ids()
        if action_space:
            if random.random() < self.__epsilon:
                action = random.choice(action_space)
            else:
                features = torch.tensor(state.process()).to(self.device)
                outputs = self.__forward(features)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_space).to(self.device))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action
