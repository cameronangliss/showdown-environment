from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from logging import Logger

import requests
from torch import Tensor

from showdown_environment.showdown.client import Client, MessageType
from showdown_environment.state.battle import Battle


class BasePlayer(Client):
    username: str
    password: str
    logger: Logger
    room: str | None

    def __init__(self, username: str, password: str):
        super().__init__(username)
        self.password = password

    @abstractmethod
    def get_action(self, state: Battle) -> int | None:
        pass

    @abstractmethod
    def encode_battle(self, battle: Battle) -> Tensor:
        pass

    ###################################################################################################################
    # OpenAI Gym-style methods

    async def setup(self):
        self.room = None
        await self.connect()
        await self.login()
        await self.forfeit_games()

    async def close(self):
        await self.leave()
        await self.logout()

    ###################################################################################################################
    # Commands to be used by Player when communicating with PokemonShowdown website

    async def login(self):
        split_message = await self.find_message(MessageType.LOGIN)
        client_id = split_message[2]
        challstr = split_message[3]
        response = requests.post(
            "https://play.pokemonshowdown.com/api/login",
            {
                "name": self.username,
                "pass": self.password,
                "challstr": f"{client_id}|{challstr}",
            },
        )
        response_json = json.loads(response.text[1:])
        assertion = response_json.get("assertion")
        await self.send_message(f"/trn {self.username},0,{assertion}")

    async def forfeit_games(self):
        # The first games message is always empty, so this is here to pass by that message.
        await self.find_message(MessageType.GAMES)
        try:
            split_message = await asyncio.wait_for(self.find_message(MessageType.GAMES), timeout=5)
        except asyncio.exceptions.TimeoutError:
            self.logger.info(
                "Second updatesearch message not received. This should mean the user just logged in."
            )
        else:
            games = json.loads(split_message[2])["games"]
            if games:
                battle_rooms = list(games.keys())
                prev_room = self.room
                for room in battle_rooms:
                    await self.join(room)
                    await self.send_message("/forfeit")
                    await self.leave()
                if prev_room:
                    await self.join(prev_room)

    async def set_avatar(self, avatar: str):
        await self.send_message(f"/avatar {avatar}")

    async def challenge(self, opponent: BasePlayer, battle_format: str, team: str | None = None):
        await self.send_message(f"/utm {team}")
        await self.send_message(f"/challenge {opponent.username}, {battle_format}")
        # Waiting for confirmation that challenge was sent
        await self.find_message(MessageType.CHALLENGE)

    async def cancel(self, opponent: BasePlayer):
        await self.send_message(f"/cancelchallenge {opponent.username}")
        # Waiting for confirmation that challenge was cancelled
        await self.find_message(MessageType.CANCEL)

    async def accept(self, opponent: BasePlayer, team: str | None = None) -> str:
        # Waiting for confirmation that challenge was received
        await self.find_message(MessageType.ACCEPT)
        await self.send_message(f"/utm {team}")
        await self.send_message(f"/accept {opponent.username}")
        # The first games message is always empty, so this is here to pass by that message.
        await self.find_message(MessageType.GAMES)
        split_message = await self.find_message(MessageType.GAMES)
        games = json.loads(split_message[2])["games"]
        room = list(games.keys())[0]
        return room

    async def join(self, room: str):
        await self.send_message(f"/join {room}")
        self.room = room

    async def timer_on(self):
        await self.send_message("/timer on")

    async def observe(self, state: Battle | None = None) -> Battle:
        split_message = await self.find_message(MessageType.OBSERVE)
        if split_message[1] == "request":
            request = json.loads(split_message[2])
            protocol = await self.find_message(MessageType.OBSERVE)
        else:
            request = None
            protocol = split_message
        if state:
            state.update(protocol, request)
        else:
            state = Battle(protocol, request)
        self.logger.info(state.get_json_str())
        return state

    async def choose(self, action: int | None, rqid: int):
        action_space = (
            [f"switch {i}" for i in range(1, 7)]
            + [f"move {i}" for i in range(1, 5)]
            + [f"move {i} mega" for i in range(1, 5)]
            + [f"move {i} zmove" for i in range(1, 5)]
            + [f"move {i} max" for i in range(1, 5)]
            + [f"move {i} terastallize" for i in range(1, 5)]
        )
        if action is not None:
            await self.send_message(f"/choose {action_space[action]}|{rqid}")

    async def leave(self):
        if self.room:
            await self.send_message(f"/leave {self.room}")
            # gets rid of all messages having to do with the room being left
            await self.find_message(MessageType.LEAVE)
            self.room = None

    async def logout(self):
        await self.send_message("/logout")
