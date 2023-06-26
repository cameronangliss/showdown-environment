from __future__ import annotations

import asyncio
import json
import logging

import requests
from state import State

from client import Client, MessageType


class Player:
    username: str
    _password: str
    _logger: logging.Logger
    _client: Client
    _room: str | None

    def __init__(self, username: str, password: str):
        # assigning other properties
        self.username = username
        self._password = password
        self._logger = logging.getLogger(username)
        self._client = Client(username, self._logger)
        self._room = None

    ###################################################################################################################
    # OpenAI Gym-style methods

    async def setup(self):
        self._room = None
        await self._client.connect()
        await self.login()
        await self.forfeit_games()

    async def close(self):
        await self.leave()
        await self.logout()

    ###################################################################################################################
    # Commands to be used by Player when communicating with PokemonShowdown website

    async def login(self):
        split_message = await self._client.find_message(self._room, MessageType.LOGIN)
        client_id = split_message[2]
        challstr = split_message[3]
        response = requests.post(
            "https://play.pokemonshowdown.com/api/login",
            {
                "name": self.username,
                "pass": self._password,
                "challstr": f"{client_id}|{challstr}",
            },
        )
        response_json = json.loads(response.text[1:])
        assertion = response_json.get("assertion")
        await self._client.send_message(self._room, f"/trn {self.username},0,{assertion}")

    async def forfeit_games(self):
        # The first games message is always empty, so this is here to pass by that message.
        await self._client.find_message(self._room, MessageType.GAMES)
        try:
            split_message = await asyncio.wait_for(self._client.find_message(self._room, MessageType.GAMES), timeout=5)
        except TimeoutError:
            self._logger.info("Second updatesearch message not received. This should mean the user just logged in.")
        else:
            games = json.loads(split_message[2])["games"]
            if games:
                battle_rooms = list(games.keys())
                prev_room = self._room
                for room in battle_rooms:
                    await self.join(room)
                    await self._client.send_message(self._room, "/forfeit")
                    await self.leave()
                if prev_room:
                    await self.join(prev_room)

    async def set_avatar(self, avatar: str):
        await self._client.send_message(self._room, f"/avatar {avatar}")

    async def challenge(self, opponent: Player, battle_format: str, team: str | None = None):
        await self._client.send_message(self._room, f"/utm {team}")
        await self._client.send_message(self._room, f"/challenge {opponent.username}, {battle_format}")
        # Waiting for confirmation that challenge was sent
        await self._client.find_message(self._room, MessageType.CHALLENGE)

    async def cancel(self, opponent: Player):
        await self._client.send_message(self._room, f"/cancelchallenge {opponent.username}")
        # Waiting for confirmation that challenge was cancelled
        await self._client.find_message(self._room, MessageType.CANCEL)

    async def accept(self, opponent: Player, team: str | None = None) -> str:
        # Waiting for confirmation that challenge was received
        await self._client.find_message(self._room, MessageType.ACCEPT)
        await self._client.send_message(self._room, f"/utm {team}")
        await self._client.send_message(self._room, f"/accept {opponent.username}")
        # The first games message is always empty, so this is here to pass by that message.
        await self._client.find_message(self._room, MessageType.GAMES)
        split_message = await self._client.find_message(self._room, MessageType.GAMES)
        games = json.loads(split_message[2])["games"]
        room = list(games.keys())[0]
        return room

    async def join(self, room: str):
        await self._client.send_message(self._room, f"/join {room}")
        self._room = room

    async def timer_on(self):
        await self._client.send_message(self._room, "/timer on")

    async def observe(self, state: State | None = None) -> State:
        split_message = await self._client.find_message(self._room, MessageType.OBSERVE)
        if split_message[1] == "request":
            request = json.loads(split_message[2])
            protocol = await self._client.find_message(self._room, MessageType.OBSERVE)
        else:
            request = None
            protocol = split_message
        if state:
            state.update(protocol, request)
        else:
            state = State(protocol, request)
        self._logger.info(state.get_json_str())
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
            await self._client.send_message(self._room, f"/choose {action_space[action]}|{rqid}")

    async def leave(self):
        if self._room:
            await self._client.send_message(self._room, f"/leave {self._room}")
            # gets rid of all messages having to do with the room being left
            await self._client.find_message(self._room, MessageType.LEAVE)
            self._room = None

    async def logout(self):
        await self._client.send_message(self._room, "/logout")
