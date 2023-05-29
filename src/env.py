import asyncio
from dataclasses import dataclass
from logging import Logger

from player import Player, PopupError
from states.state import State


@dataclass
class Env:
    _player1: Player
    _player2: Player
    logger: Logger

    ###################################################################################################################
    # OpenAI Gym-style methods

    async def setup(self):
        await self._player1.setup()
        await self._player2.setup()

    async def reset(self, format_str: str, avatar1: str, avatar2: str) -> tuple[State, State]:
        await self._player1.leave()
        await self._player2.leave()
        await self._player1.set_avatar(avatar1)
        await self._player2.set_avatar(avatar2)
        while True:
            try:
                await self._player1.challenge(self._player2, format_str)
                room = await self._player2.accept(self._player1)
                break
            except PopupError as e1:
                self.logger.warning(e1)
                try:
                    await self._player1.cancel(self._player2)
                except PopupError as e2:
                    self.logger.warning(e2)
                await asyncio.sleep(5)
        await self._player1.join(room)
        await self._player2.join(room)
        await self._player1.timer_on()
        state1 = await self._player1.observe()
        state2 = await self._player2.observe()
        return state1, state2

    async def step(
        self,
        state1: State,
        state2: State,
        action1: int | None,
        action2: int | None,
    ) -> tuple[State, State, int, int, bool]:
        await self._player1.choose(action1, state1.request["rqid"])
        await self._player2.choose(action2, state2.request["rqid"])
        new_state1 = await self._player1.observe(state1)
        new_state2 = await self._player2.observe(state2)
        done = "win" in new_state1.protocol or "tie" in new_state1.protocol
        reward1, reward2 = self.__get_rewards(new_state1)
        return new_state1, new_state2, reward1, reward2, done

    async def close(self):
        await self._player1.close()
        await self._player2.close()

    ###################################################################################################################
    # Helper methods

    def __get_rewards(self, state: State) -> tuple[int, int]:
        if "win" in state.protocol:
            i = state.protocol.index("win")
            winner = state.protocol[i + 1].strip()
            if winner == self._player1.username:
                return 1, -1
            else:
                return -1, 1
        elif "tie" in state.protocol:
            return 0, 0
        else:
            return 0, 0
