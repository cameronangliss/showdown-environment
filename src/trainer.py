import random
from dataclasses import dataclass
from datetime import datetime

from websockets.exceptions import ConnectionClosedError

from env import Env
from model import Model
from observation import Observation


@dataclass
class Trainer:
    model: Model
    env: Env

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

    async def run_episode(self, format_str: str) -> str | None:
        try:
            obs1, obs2 = await self.env.reset(format_str)
            done = False
            while not done:
                action1 = self.model.get_action(obs1)
                action2 = self.model.get_action(obs2)
                next_obs1, next_obs2, done = await self.env.step(
                    action1,
                    action2,
                    obs1.request["rqid"],
                    obs2.request["rqid"],
                )
                reward1, reward2 = self.get_rewards(next_obs1)
                self.model.update(obs1, action1, reward1, next_obs1, done)
                self.model.update(obs2, action2, reward2, next_obs2, done)
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
        random_formats = [f"gen{n}randombattle" for n in range(1, 10)]
        for i in range(num_episodes):
            winner = await self.run_episode(random.choice(random_formats))
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: {winner} wins game {i + 1}")
        await self.env.close()
