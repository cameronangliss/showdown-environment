from __future__ import annotations

import json
import random
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from encoders import encode_battle
from memory import Experience, Memory
from torch import Tensor
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from showdown_environment.showdown.environment import Environment
from showdown_environment.state.battle import Battle


class Model(nn.Module):
    __alpha: float
    __epsilon: float
    __gamma: float
    __memory_length: int
    __layers: nn.Sequential

    def __init__(self, alpha: float, epsilon: float, gamma: float, memory_length: int, hidden_layer_sizes: list[int]):
        super().__init__()  # type: ignore
        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__gamma = gamma
        self.__memory_length = memory_length
        self.memory = Memory([], maxlen=memory_length)
        layer_sizes = [1504, *hidden_layer_sizes, 26]
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()]
        self.__layers = nn.Sequential(*layers[:-1])
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.__alpha, momentum=0.9, weight_decay=1e-4)
        # Move the model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def __forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.__layers(x)

    def __update(self, experience: Experience):
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
        duplicate_model = deepcopy(self)
        print("Generating experiences...")
        experiences, _ = await self.__run_episodes(duplicate_model, 100)
        self.memory.extend(experiences)
        print("Done!")
        while True:
            print(f"Training on {len(self.memory)} experiences...")
            for i in range(1000):
                batch = self.memory.sample(round(len(self.memory) / 100))
                for exp in batch:
                    self.__update(exp)
                print(f"Progress: {(i + 1) / 10}%", end="\r")
            print("Done!                         ")
            print("Evaluating model...")
            experiences, num_wins = await self.__run_episodes(duplicate_model, 100, 0.55)
            if num_wins < 55:
                print("Improvement failed.")
                self.memory.extend(experiences)
            else:
                print("Improvement succeeded!")
                self.memory.clear()
                break

    async def __run_episodes(
        self, alt_model: Model, num_episodes: int, min_win_rate: float | None = None
    ) -> tuple[list[Experience], float]:
        with open("config.json") as f:
            config = json.load(f)
        env = Environment(config["username"], config["password"], config["alt_username"], config["alt_password"])
        await env.setup()
        experiences: list[Experience] = []
        num_wins = 0
        num_iters = 0
        time = datetime.now().strftime("%H:%M:%S")
        for i in range(num_episodes):
            new_experiences, winner = await self.__run_episode(alt_model, env, "gen4randombattle")
            if len(experiences) + len(new_experiences) > self.__memory_length:
                break
            experiences += new_experiences
            if winner is None:
                num_wins += 0.5
            elif winner == env.agent.username:
                num_wins += 1
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: Win rate = {num_wins}/{i + 1}", end="\r")
            num_iters += 1
            if min_win_rate is not None and (
                num_wins / num_episodes >= min_win_rate or (i - num_wins) / num_episodes > 1 - min_win_rate
            ):
                break
        print(f"{time}: Win rate = {num_wins}/{num_iters}")
        await env.close()
        return experiences, num_wins

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
                    encode_battle(state1).to(self.device),
                    action1,
                    encode_battle(next_state1).to(self.device),
                    reward1,
                    done,
                )
                experience2 = Experience(
                    encode_battle(state2).to(self.device),
                    action2,
                    encode_battle(next_state2).to(self.device),
                    reward2,
                    done,
                )
                experiences += [experience1, experience2]
                state1, state2 = next_state1, next_state2
            try:
                winner_id = state1.protocol.index("win")
            except ValueError:
                winner = None
            else:
                winner = state1.protocol[winner_id + 1].strip()
            meaningful_experiences = list(filter(lambda experience: experience.action is not None, experiences))
            return meaningful_experiences, winner
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
                features = encode_battle(state).to(self.device)
                outputs = self.__forward(features)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_space).to(self.device))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action
