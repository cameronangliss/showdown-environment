import asyncio
import random
from dataclasses import dataclass
from logging import Logger

from player import Player, PopupError
from states.state import State


@dataclass
class Env:
    player: Player
    _alt_player: Player
    logger: Logger
    __cynthia_avatars = [
        "cynthia-anime",
        "cynthia-anime2",
        "cynthia-gen4",
        "cynthia-gen7",
        "cynthia-masters",
        "cynthia-masters2",
        "cynthia-masters3",
        "cynthia",
    ]

    ###################################################################################################################
    # OpenAI Gym-style methods

    async def setup(self):
        await self.player.setup()
        await self._alt_player.setup()

    async def reset(self, format_str: str) -> tuple[State, State]:
        await self.player.leave()
        await self._alt_player.leave()
        [avatar1, avatar2, *_] = random.sample(self.__cynthia_avatars, k=2)
        await self.player.set_avatar(avatar1)
        await self._alt_player.set_avatar(avatar2)
        while True:
            try:
                await self.player.challenge(self._alt_player, format_str)
                room = await self._alt_player.accept(self.player)
                break
            except PopupError as e1:
                self.logger.warning(e1)
                try:
                    await self.player.cancel(self._alt_player)
                except PopupError as e2:
                    self.logger.warning(e2)
                await asyncio.sleep(5)
        await self.player.join(room)
        await self._alt_player.join(room)
        await self.player.timer_on()
        state1 = await self.player.observe()
        state2 = await self._alt_player.observe()
        return state1, state2

    async def step(
        self,
        state1: State,
        state2: State,
        action1: int | None,
        action2: int | None,
    ) -> tuple[State, State, int, bool]:
        await self.player.choose(action1, state1.request["rqid"])
        await self._alt_player.choose(action2, state2.request["rqid"])
        new_state1 = await self.player.observe(state1)
        new_state2 = await self._alt_player.observe(state2)
        done = "win" in new_state1.protocol or "tie" in new_state1.protocol
        reward = self.__get_reward(new_state1)
        return new_state1, new_state2, reward, done

    async def close(self):
        await self.player.close()
        await self._alt_player.close()

    ###################################################################################################################
    # Helper methods

    def __get_reward(self, state: State) -> int:
        if "win" in state.protocol:
            i = state.protocol.index("win")
            winner = state.protocol[i + 1].strip()
            if winner == self.player.username:
                return 1
            else:
                return -1
        elif "tie" in state.protocol:
            return 0
        else:
            return 0
