import asyncio
from dataclasses import dataclass
from logging import Logger
from websockets.exceptions import ConnectionClosedError

from player import Player


@dataclass
class Battle:
    player1: Player
    player2: Player
    logger: Logger

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
                await self.player1.cancel(self.player2)
                await asyncio.sleep(5)
        room = await self.player2.accept(self.player1)
        await self.player1.join(room)
        await self.player2.join(room)
        await self.player1.timer_on()

    async def close(self):
        await self.player1.logout()
        await self.player2.logout()

    async def run_episode(self) -> str:
        while True:
            try:
                await self.reset()
                while True:
                    try:
                        await self.player1.step()
                        await self.player2.step()
                    except SystemExit as e:
                        await self.player1.leave()
                        await self.player2.leave()
                        return str(e).strip()
            except ConnectionClosedError:
                self.logger.error("Connection closed unexpectedly")
                await self.setup()
