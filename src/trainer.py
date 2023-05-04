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

    def get_action(self, obs: Observation) -> tuple[int, str] | tuple[None, None]:
        action_space = Player.get_action_space(obs)
        if action_space:
            if random.random() < self.epsilon:
                action = random.choice(action_space)
            else:
                action_ids = list(map(lambda action: action[0], action_space))
                outputs = self.model(obs)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_ids))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action
        else:
            return None, None

    def get_rewards(self, obs: Observation) -> tuple[int, int]:
        if "win" in obs.protocol:
            i = obs.protocol.index("win")
            winner = obs.protocol[i + 1].strip()
            if winner == self.env.player1.username:
                return 1, -1
            else:
                return -1, 1
        elif "tie" in obs.protocol:
            return 0, 0
        else:
            return 0, 0

    def update_model(self, obs: Observation, action: int | None, reward: int, next_obs: Observation, done: bool):
        if action:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            if done:
                q_target = torch.tensor(reward)
            else:
                next_q_values = self.model(next_obs)
                q_target = reward + self.gamma * torch.max(next_q_values)
            q_values = self.model(obs)
            q_estimate = q_values[action]
            td_error = q_target - q_estimate
            loss = td_error**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    async def run_episode(self) -> str | None:
        try:
            obs1, obs2 = await self.env.reset()
            done = False
            while not done:
                action1_id, action1_str = self.get_action(obs1)
                action2_id, action2_str = self.get_action(obs2)
                next_obs1, next_obs2, done = await self.env.step(
                    action1_str, action2_str, obs1.request["rqid"], obs2.request["rqid"]
                )
                reward1, reward2 = self.get_rewards(next_obs1)
                self.update_model(obs1, action1_id, reward1, next_obs1, done)
                self.update_model(obs2, action2_id, reward2, next_obs2, done)
                obs1, obs2 = next_obs1, next_obs2
            try:
                winner_id = obs1.protocol.index("win")
            except ValueError:
                winner = None
            else:
                winner = obs1.protocol[winner_id + 1].strip()
            return winner
        except ConnectionClosedError:
            self.env.logger.error("Connection closed unexpectedly")
            await self.env.setup()
            winner = None
            return winner

    async def train(self, num_episodes: int):
        await self.env.setup()
        for i in range(num_episodes):
            winner = await self.run_episode()
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: {winner} wins game {i + 1}")
        await self.env.close()
