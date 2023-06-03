import asyncio
from dataclasses import dataclass
from logging import Logger

from player import Player, PopupError
from states.state import State


@dataclass
class Env:
    _player: Player
    _alt_player: Player
    logger: Logger

    ###################################################################################################################
    # OpenAI Gym-style methods

    async def setup(self):
        await self._player.setup()
        await self._alt_player.setup()

    async def reset(self, format_str: str, avatar1: str, avatar2: str) -> tuple[State, State]:
        await self._player.leave()
        await self._alt_player.leave()
        await self._player.set_avatar(avatar1)
        await self._alt_player.set_avatar(avatar2)
        while True:
            try:
                await self._player.challenge(self._alt_player, format_str)
                room = await self._alt_player.accept(self._player)
                break
            except PopupError as e1:
                self.logger.warning(e1)
                try:
                    await self._player.cancel(self._alt_player)
                except PopupError as e2:
                    self.logger.warning(e2)
                await asyncio.sleep(5)
        await self._player.join(room)
        await self._alt_player.join(room)
        await self._player.timer_on()
        state1 = await self._player.observe()
        state2 = await self._alt_player.observe()
        return state1, state2

    async def step(
        self,
        state1: State,
        state2: State,
        action1: int | None,
        action2: int | None,
    ) -> tuple[State, State, int, int, bool]:
        await self._player.choose(action1, state1.request["rqid"])
        await self._alt_player.choose(action2, state2.request["rqid"])
        new_state1 = await self._player.observe(state1)
        new_state2 = await self._alt_player.observe(state2)
        done = "win" in new_state1.protocol or "tie" in new_state1.protocol
        reward1, reward2 = self.__get_rewards(new_state1)
        return new_state1, new_state2, reward1, reward2, done

    async def close(self):
        await self._player.close()
        await self._alt_player.close()

    ###################################################################################################################
    # Helper methods

    def __get_rewards(self, state: State) -> tuple[int, int]:
        if "win" in state.protocol:
            i = state.protocol.index("win")
            winner = state.protocol[i + 1].strip()
            if winner == self._player.username:
                return 1, -1
            else:
                return -1, 1
        elif "tie" in state.protocol:
            return 0, 0
        else:
            return 0, 0
