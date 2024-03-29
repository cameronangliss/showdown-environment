import asyncio
import logging
import random

from showdown_environment.showdown.client import PopupError
from showdown_environment.showdown.player import Player
from showdown_environment.state.battle import Battle


class Environment:
    agent: Player
    __alt_agent: Player
    logger: logging.Logger

    def __init__(self, username: str, password: str, alt_username: str, alt_password: str):
        # building logger
        logging.basicConfig(
            filename="debug.log",
            filemode="w",
            format="%(name)s %(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s()\n%(message)s\n",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("Environment")

        # construct players
        self.agent = Player(username, password)
        self.__alt_agent = Player(alt_username, alt_password)

    ###################################################################################################################
    # OpenAI Gym-style methods

    async def setup(self):
        await self.agent.setup()
        await self.__alt_agent.setup()

    async def reset(self, format_str: str) -> tuple[Battle, Battle]:
        await self.agent.leave()
        await self.__alt_agent.leave()
        cynthia_avatars = [
            "cynthia-anime",
            "cynthia-anime2",
            "cynthia-gen4",
            "cynthia-gen7",
            "cynthia-masters",
            "cynthia-masters2",
            "cynthia-masters3",
            "cynthia",
        ]
        [avatar1, avatar2, *_] = random.sample(cynthia_avatars, k=2)
        await self.agent.set_avatar(avatar1)
        await self.__alt_agent.set_avatar(avatar2)
        while True:
            try:
                await self.agent.challenge(self.__alt_agent, format_str)
                room = await self.__alt_agent.accept(self.agent)
                break
            except PopupError as e1:
                self.logger.warning(e1)
                try:
                    await self.agent.cancel(self.__alt_agent)
                except PopupError as e2:
                    self.logger.warning(e2)
                if "Due to spam from your internet provider, you can't challenge others right now." in str(e1):
                    self.logger.info("Waiting for 5 hours to be allowed back in...")
                    await asyncio.sleep(5 * 60 * 60)
                else:
                    await asyncio.sleep(5)
        await self.agent.join(room)
        await self.__alt_agent.join(room)
        await self.agent.timer_on()
        state1 = await self.agent.observe()
        state2 = await self.__alt_agent.observe()
        return state1, state2

    async def step(
        self,
        state1: Battle,
        state2: Battle,
        action1: int | None,
        action2: int | None,
    ) -> tuple[Battle, Battle, int, int, bool]:
        await self.agent.choose(action1, state1.request["rqid"])
        await self.__alt_agent.choose(action2, state2.request["rqid"])
        next_state1 = await self.agent.observe(state1)
        next_state2 = await self.__alt_agent.observe(state2)
        reward1, reward2 = self.__get_rewards(next_state1)
        done = "win" in next_state1.protocol or "tie" in next_state1.protocol
        return next_state1, next_state2, reward1, reward2, done

    async def close(self):
        await self.agent.close()
        await self.__alt_agent.close()

        # resetting logger
        for handler in logging.getLogger().handlers:
            logging.getLogger().removeHandler(handler)

    ###################################################################################################################
    # Helper methods

    def __get_rewards(self, state: Battle) -> tuple[int, int]:
        if "win" in state.protocol:
            i = state.protocol.index("win")
            winner = state.protocol[i + 1].strip()
            if winner == self.agent.username:
                return 1, -1
            else:
                return -1, 1
        elif "tie" in state.protocol:
            return 0, 0
        else:
            return 0, 0
