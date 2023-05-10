import asyncio
import random
from dataclasses import dataclass
from logging import Logger

from observation import Observation
from player import Player


@dataclass
class Env:
    player1: Player
    player2: Player
    logger: Logger
    avatars = [
        "cynthia-anime",
        "cynthia-anime2",
        "cynthia-gen4",
        "cynthia-gen7",
        "cynthia-masters",
        "cynthia-masters2",
        "cynthia-masters3",
        "cynthia",
    ]

    async def setup(self):
        self.player1.room = None
        self.player2.room = None
        await self.player1.connect()
        await self.player2.connect()
        await self.player1.login()
        await self.player2.login()
        await self.player1.forfeit_games()
        await self.player2.forfeit_games()

    async def reset(self, format_str: str) -> tuple[Observation, Observation]:
        await self.player1.leave()
        await self.player2.leave()
        await self.player1.set_avatar(random.choice(self.avatars))
        await self.player2.set_avatar(random.choice(self.avatars))
        while True:
            try:
                await self.player1.challenge(self.player2, format_str)
                break
            except RuntimeError as e:
                self.logger.warning(e)
                if "You are already challenging someone" in str(e):
                    await self.player1.cancel(self.player2)
                await asyncio.sleep(5)
        room = await self.player2.accept(self.player1)
        await self.player1.join(room)
        await self.player2.join(room)
        await self.player1.timer_on()
        obs1 = await self.player1.observe()
        obs2 = await self.player2.observe()
        return obs1, obs2

    async def step(
        self, action1: int | None, action2: int | None, rqid1: int, rqid2: int
    ) -> tuple[Observation, Observation, int, int, bool]:
        await self.player1.choose(action1, rqid1)
        await self.player2.choose(action2, rqid2)
        obs1 = await self.player1.observe()
        obs2 = await self.player2.observe()
        done = "win" in obs1.protocol or "tie" in obs1.protocol
        reward1, reward2 = self.__get_rewards(obs1)
        return obs1, obs2, reward1, reward2, done

    def __get_rewards(self, obs: Observation) -> tuple[int, int]:
        if "win" in obs.protocol:
            i = obs.protocol.index("win")
            winner = obs.protocol[i + 1].strip()
            if winner == self.player1.username:
                return 1, -1
            else:
                return -1, 1
        elif "tie" in obs.protocol:
            return 0, 0
        else:
            return 0, 0

    async def close(self):
        await self.player1.leave()
        await self.player2.leave()
        await self.player1.logout()
        await self.player2.logout()
