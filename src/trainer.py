from dataclasses import dataclass
from datetime import datetime
import random
import torch
from torch import Tensor
from websockets.exceptions import ConnectionClosedError

from model import Model
from env import Env
from player import Observation


@dataclass
class Trainer:
    model: Model
    env: Env
    gamma: float = 0.9  # Discount factor
    alpha: float = 0.01  # Learning rate
    epsilon: float = 0.1  # Exploration rate

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
        action_space = Trainer.get_valid_action_ids(obs)
        if action_space:
            if random.random() < self.epsilon:
                action = random.choice(action_space)
            else:
                outputs = self.model(obs)
                valid_outputs = torch.index_select(outputs, dim=0, index=torch.tensor(action_space))
                max_output_id = int(torch.argmax(valid_outputs).item())
                action = action_space[max_output_id]
            return action

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
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
            if done:
                q_target = torch.tensor(reward)
            else:
                next_q_values: Tensor = self.model(next_obs)
                q_target = reward + self.gamma * torch.max(next_q_values)
            q_values: Tensor = self.model(obs)
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
                action1 = self.get_action(obs1)
                action2 = self.get_action(obs2)
                next_obs1, next_obs2, done = await self.env.step(
                    action1,
                    action2,
                    obs1.request["rqid"],
                    obs2.request["rqid"],
                )
                reward1, reward2 = self.get_rewards(next_obs1)
                self.update_model(obs1, action1, reward1, next_obs1, done)
                self.update_model(obs2, action2, reward2, next_obs2, done)
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
