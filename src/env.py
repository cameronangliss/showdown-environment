import asyncio
from dataclasses import dataclass
from logging import Logger

from player import Player, Observation


@dataclass
class Env:
    player1: Player
    player2: Player
    logger: Logger

    async def setup(self):
        self.player1.room = None
        self.player2.room = None
        await self.player1.connect()
        await self.player2.connect()
        await self.player1.login("ethan")
        await self.player2.login("cynthia-gen7")
        await self.player1.forfeit_games()
        await self.player2.forfeit_games()

    async def reset(self) -> tuple[Observation, Observation]:
        await self.player1.leave()
        await self.player2.leave()
        while True:
            try:
                await self.player1.challenge(self.player2)
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
        self, action1: str | None, action2: str | None, rqid1: int, rqid2: int
    ) -> tuple[Observation, Observation, bool]:
        await self.player1.choose(action1, rqid1)
        await self.player2.choose(action2, rqid2)
        obs1 = await self.player1.observe()
        obs2 = await self.player2.observe()
        done = "win" in obs1.protocol or "tie" in obs1.protocol
        return obs1, obs2, done

    async def close(self):
        await self.player1.leave()
        await self.player2.leave()
        await self.player1.logout()
        await self.player2.logout()
