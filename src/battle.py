import asyncio
from logging import Logger
import websockets

from player import Player


class Battle:
    def __init__(self, player1: Player, player2: Player, logger: Logger):
        self.player1 = player1
        self.player2 = player2
        self.logger = logger

    async def setup(self):
        await self.player1.setup()
        await self.player2.setup()

    async def reset(self):
        while True:
            try:
                await self.player1.challenge(self.player2)
                break
            except RuntimeError as e:
                self.logger.warning(f"Popup occurred: {str(e)}")
                await asyncio.sleep(5)
        room = await self.player2.accept(self.player1)
        await self.player1.join(room)
        await self.player2.join(room)
        await self.player1.timer_on()

    async def step(self) -> tuple[bool, str]:
        done, winner = await self.player1.observe()
        if not done:
            done, winner = await self.player2.observe()
            await self.player1.choose("default")
            await self.player2.choose("default")
        return done, winner

    async def close(self):
        await self.player1.logout()
        await self.player2.logout()

    async def run_episode(self) -> str:
        while True:
            try:
                await self.reset()
                done = False
                while not done:
                    done, winner = await self.step()
                await self.player1.leave()
                await self.player2.leave()
                return winner
            except websockets.exceptions.ConnectionClosedError:
                self.logger.error("Connection closed unexpectedly")
                await self.setup()
