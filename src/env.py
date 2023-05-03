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
        await self.player1.setup()
        await self.player2.setup()

    async def reset(self) -> tuple[Observation, Observation]:
        while True:
            try:
                await self.player1.challenge(self.player2)
                break
            except RuntimeError as e:
                self.logger.warning(e)
                if (
                    str(e)
                    == "You are already challenging someone. Cancel that challenge before challenging someone else."
                ):
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
        await self.player1.logout()
        await self.player2.logout()
