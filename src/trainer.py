from dataclasses import dataclass
from datetime import datetime
import random
import torch
from websockets.exceptions import ConnectionClosedError

from model import Model
from env import Env
from player import Player, Observation


@dataclass
class Trainer:
    model: Model
    env: Env
    gamma: float = 0.9  # Discount factor
    alpha: float = 0.1  # Learning rate
    epsilon: float = 0.1  # Exploration rate

    # Define the Q-learning update function
    def q_learning_update(self, state, action, reward, next_state, done):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        # Compute the Q-value target
        if done:
            q_target = reward
        else:
            next_q_values = self.model(next_state)
            q_target = reward + self.gamma * torch.max(next_q_values)
        # Compute the Q-value estimate
        q_values = self.model(state)
        q_estimate = q_values[action]
        # Compute the TD error and update the Q-value estimate
        td_error = q_target - q_estimate
        loss = td_error**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def get_action(self, obs: Observation) -> str | None:
        action_space = Player.get_action_space(obs)
        if action_space:
            if random.random() < self.epsilon:
                _, action = random.choice(action_space)
            else:
                action_ids = list(map(lambda tuple: tuple[0], action_space))
                actions = list(map(lambda tuple: tuple[1], action_space))
                outputs = self.model(obs)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_ids))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = actions[max_output_id]
            return action

    async def run_episode(self) -> str:
        try:
            obs1, obs2 = await self.env.reset()
            done = False
            while not done:
                action1 = self.get_action(obs1)
                action2 = self.get_action(obs2)
                obs1, obs2, done = await self.env.step(action1, action2, obs1.request["rqid"], obs2.request["rqid"])
                # self.q_learning_update(state, action, reward, next_state, done)
            try:
                winner_id = obs1.protocol.index("win")
            except ValueError:
                winner = "None"
            else:
                winner = obs1.protocol[winner_id + 1].strip()
            return winner
        except ConnectionClosedError:
            self.env.logger.error("Connection closed unexpectedly")
            await self.env.setup()
            return "None"

    async def train(self, num_episodes: int):
        await self.env.setup()
        for i in range(num_episodes):
            winner = await self.run_episode()
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: {winner} wins game {i + 1}")
        await self.env.close()
