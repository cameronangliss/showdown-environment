import asyncio
import logging
import random
from datetime import datetime

from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from showdown_environment.showdown.base_player import BasePlayer
from showdown_environment.showdown.client import PopupError
from showdown_environment.showdown.experience import Experience
from showdown_environment.state.battle import Battle


class Environment:
    player: BasePlayer
    logger: logging.Logger

    def __init__(self, player: BasePlayer):
        self.player = player
        logging.basicConfig(
            filename="debug.log",
            filemode="w",
            format="%(name)s %(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s()\n%(message)s\n",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("Environment")

    async def run_episodes(
        self,
        player: BasePlayer,
        num_episodes: int,
        min_win_rate: float | None = None,
        memory_length: int | None = None,
    ) -> tuple[list[Experience], float]:
        await self.setup(player)
        experiences: list[Experience] = []
        num_wins = 0
        num_iters = 0
        time = datetime.now().strftime("%H:%M:%S")
        for i in range(num_episodes):
            new_experiences, winner = await self.__run_episode(player, "gen4randombattle")
            if (
                memory_length is not None
                and len(experiences) + len(new_experiences) > memory_length
            ):
                break
            experiences += new_experiences
            if winner is None:
                num_wins += 0.5
            elif winner == player.username:
                num_wins += 1
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time}: Win rate = {num_wins}/{i + 1}", end="\r")
            num_iters += 1
            if min_win_rate is not None and (
                num_wins >= round(min_win_rate * num_episodes, 5)
                or num_iters - num_wins > round((1 - min_win_rate) * num_episodes, 5)
            ):
                break
        print(f"{time}: Win rate = {num_wins}/{num_iters}")
        await self.close(player)
        return experiences, num_wins

    async def __run_episode(
        self, player: BasePlayer, format_str: str
    ) -> tuple[list[Experience], str | None]:
        experiences: list[Experience] = []
        try:
            state1, state2 = await self.reset(player, format_str)
            turn = 0
            done = False
            while not done:
                turn += 1
                action1 = player.get_action(state1)
                action2 = self.player.get_action(state2)
                next_state1, next_state2, reward1, reward2, done = await self.step(
                    player,
                    state1,
                    state2,
                    action1,
                    action2,
                )
                experience1 = Experience(
                    player.encode_battle(state1),
                    action1,
                    player.encode_battle(next_state1),
                    reward1,
                    done,
                )
                experience2 = Experience(
                    self.player.encode_battle(state2),
                    action2,
                    self.player.encode_battle(next_state2),
                    reward2,
                    done,
                )
                experiences += [experience1, experience2]
                state1, state2 = next_state1, next_state2
            try:
                winner_id = state1.protocol.index("win")
            except ValueError:
                winner = None
            else:
                winner = state1.protocol[winner_id + 1].strip()
            meaningful_experiences = list(
                filter(lambda experience: experience.action is not None, experiences)
            )
            return meaningful_experiences, winner
        except (ConnectionClosedError, ConnectionClosedOK):
            self.logger.error("Connection closed unexpectedly")
            await self.setup(player)
            winner = None
            return [], winner

    ###################################################################################################################
    # OpenAI Gym-style methods

    async def setup(self, player: BasePlayer):
        await player.setup()
        await self.player.setup()

    async def reset(self, player: BasePlayer, format_str: str) -> tuple[Battle, Battle]:
        await self.player.leave()
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
        await player.set_avatar(avatar1)
        await self.player.set_avatar(avatar2)
        while True:
            try:
                await player.challenge(self.player, format_str)
                room = await self.player.accept(player)
                break
            except PopupError as e1:
                self.logger.warning(e1)
                try:
                    await player.cancel(self.player)
                except PopupError as e2:
                    self.logger.warning(e2)
                if (
                    "Due to spam from your internet provider, you can't challenge others right now."
                    in str(e1)
                ):
                    self.logger.info("Waiting for 5 hours to be allowed back in...")
                    await asyncio.sleep(5 * 60 * 60)
                else:
                    await asyncio.sleep(5)
        await player.join(room)
        await self.player.join(room)
        await player.timer_on()
        state1 = await player.observe()
        state2 = await self.player.observe()
        return state1, state2

    async def step(
        self,
        player: BasePlayer,
        state1: Battle,
        state2: Battle,
        action1: int | None,
        action2: int | None,
    ) -> tuple[Battle, Battle, int, int, bool]:
        await player.choose(action1, state1.request["rqid"])
        await self.player.choose(action2, state2.request["rqid"])
        next_state1 = await player.observe(state1)
        next_state2 = await self.player.observe(state2)
        reward1, reward2 = self.__get_rewards(next_state1)
        done = "win" in next_state1.protocol or "tie" in next_state1.protocol
        return next_state1, next_state2, reward1, reward2, done

    async def close(self, player: BasePlayer):
        await player.close()
        await self.player.close()

        # resetting logger
        for handler in logging.getLogger().handlers:
            logging.getLogger().removeHandler(handler)

    ###################################################################################################################
    # Helper methods

    def __get_rewards(self, state: Battle) -> tuple[int, int]:
        if "win" in state.protocol:
            i = state.protocol.index("win")
            winner = state.protocol[i + 1].strip()
            if winner == self.player.username:
                return -1, 1
            else:
                return 1, -1
        elif "tie" in state.protocol:
            return 0, 0
        else:
            return 0, 0
