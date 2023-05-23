import asyncio
import random
from dataclasses import dataclass
from logging import Logger

from player import Player, PopupError
from states.state import State


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

    async def reset(self, format_str: str) -> tuple[State, State]:
        await self.player1.leave()
        await self.player2.leave()
        await self.player1.set_avatar(random.choice(self.avatars))
        await self.player2.set_avatar(random.choice(self.avatars))
        while True:
            try:
                await self.player1.challenge(self.player2, format_str)
                room = await self.player2.accept(self.player1)
                break
            except PopupError as e1:
                self.logger.warning(e1)
                try:
                    await self.player1.cancel(self.player2)
                except PopupError as e2:
                    self.logger.warning(e2)
                await asyncio.sleep(5)
        await self.player1.join(room)
        await self.player2.join(room)
        await self.player1.timer_on()
        state1 = await self.player1.observe()
        state2 = await self.player2.observe()
        return state1, state2

    async def step(
        self,
        state1: State,
        state2: State,
        action1: int | None,
        action2: int | None,
    ) -> tuple[State, State, int, int, bool]:
        await self.player1.choose(action1, state1.request["rqid"])
        await self.player2.choose(action2, state2.request["rqid"])
        new_state1 = await self.player1.observe(state1)
        new_state2 = await self.player2.observe(state2)
        done = "win" in new_state1.protocol or "tie" in new_state1.protocol
        reward1, reward2 = self.__get_rewards(new_state1)
        return new_state1, new_state2, reward1, reward2, done

    def __get_rewards(self, state: State) -> tuple[int, int]:
        if "win" in state.protocol:
            i = state.protocol.index("win")
            winner = state.protocol[i + 1].strip()
            if winner == self.player1.username:
                return 1, -1
            else:
                return -1, 1
        elif "tie" in state.protocol:
            return 0, 0
        else:
            return 0, 0

    async def close(self):
        await self.player1.leave()
        await self.player2.leave()
        await self.player1.logout()
        await self.player2.logout()
