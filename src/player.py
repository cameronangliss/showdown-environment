from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum, auto
import json
from logging import Logger
import requests
from typing import Any, NamedTuple
import websockets.client as ws


class MessageType(Enum):
    LOGIN = auto()
    GAMES = auto()
    CHALLENGE = auto()
    CANCEL = auto()
    ACCEPT = auto()
    REQUEST = auto()
    OBSERVE = auto()
    LEAVE = auto()


class Observation(NamedTuple):
    request: Any
    split_message: list[str]


@dataclass
class Player:
    username: str
    password: str
    logger: Logger
    websocket: ws.WebSocketClientProtocol | None = None
    room: str | None = None

    async def connect(self):
        while True:
            try:
                self.websocket = await ws.connect("wss://sim3.psim.us/showdown/websocket")
                break
            except TimeoutError:
                self.logger.error("Connection attempt failed, retrying now")

    async def send_message(self, message: str):
        room = self.room or ""
        message = f"{room}|{message}"
        self.logger.info(f"{self.username.upper()} -> SERVER:\n{message}")
        if self.websocket:
            await self.websocket.send(message)
        else:
            raise RuntimeError("Cannot send message without established websocket")

    async def receive_message(self) -> str:
        if self.websocket:
            response = str(await self.websocket.recv())
        else:
            raise RuntimeError("Cannot receive message without established websocket")
        self.logger.info(f"SERVER -> {self.username.upper()}:\n{response}")
        return response

    async def find_message(self, message_type: MessageType) -> list[str]:
        while True:
            message = await self.receive_message()
            split_message = message.split("|")
            match message_type:
                case MessageType.LOGIN:
                    if split_message[1] == "challstr":
                        return split_message
                case MessageType.GAMES:
                    if split_message[1] == "updatesearch":
                        return split_message
                case MessageType.CHALLENGE:
                    if split_message[1] == "popup":
                        # too many games or challenges in too short a period of time cause a popup
                        raise RuntimeError(split_message[2])
                    elif (
                        split_message[1] == "pm"
                        and split_message[2] == f" {self.username}"
                        and "wants to battle!" in split_message[4]
                    ):
                        return split_message
                case MessageType.CANCEL:
                    if (
                        split_message[1] == "pm"
                        and split_message[2] == f" {self.username}"
                        and "cancelled the challenge." in split_message[4]
                    ):
                        return split_message
                case MessageType.ACCEPT:
                    if (
                        split_message[1] == "pm"
                        and split_message[3] == f" {self.username}"
                        and "wants to battle!" in split_message[4]
                    ):
                        return split_message
                case MessageType.REQUEST:
                    if "win" in split_message:
                        winner_index = split_message.index("win") + 1
                        winner = split_message[winner_index]
                        raise SystemExit(winner)
                    elif "tie" in split_message:
                        raise SystemExit("None")
                    elif split_message[1] == "request" and split_message[2]:
                        return split_message
                case MessageType.OBSERVE:
                    if "t:" in split_message:
                        return split_message
                case MessageType.LEAVE:
                    if self.room and self.room in split_message[0] and split_message[1] == "deinit":
                        return split_message

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
            split_message = await self.find_message(MessageType.GAMES)
        except asyncio.TimeoutError:
            self.logger.info("Second updatesearch message not received. This should mean the user just logged in.")
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

    async def challenge(self, opponent: Player, battle_format: str = "gen9randombattle", team: str | None = None):
        await self.send_message(f"/utm {team}")
        await self.send_message(f"/challenge {opponent.username}, {battle_format}")
        # Waiting for confirmation that challenge was sent
        await self.find_message(MessageType.CHALLENGE)

    async def cancel(self, opponent: Player):
        await self.send_message(f"/cancelchallenge {opponent.username}")
        # Waiting for confirmation that challenge was cancelled
        await self.find_message(MessageType.CANCEL)

    async def accept(self, opponent: Player, team: str | None = None) -> str:
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

    async def observe(self) -> Observation:
        split_message = await self.find_message(MessageType.REQUEST)
        request = json.loads(split_message[2])
        observations = await self.find_message(MessageType.OBSERVE)
        return Observation(request, observations)

    async def choose(self, action: str | None, rqid: int):
        if action:
            await self.send_message(f"/choose {action}|{rqid}")

    async def leave(self):
        await self.send_message(f"/leave {self.room}")
        # gets rid of all messages having to do with the room being left
        await self.find_message(MessageType.LEAVE)
        self.room = None
        self.request = None
        self.observations = None

    async def logout(self):
        await self.send_message("/logout")

    async def setup(self):
        self.room = None
        await self.connect()
        await self.login()
        try:
            await asyncio.wait_for(self.forfeit_games(), timeout=5)
        except TimeoutError:
            pass

    @staticmethod
    def get_action_space(obs: Observation) -> list[tuple[int, str]]:
        switch_actions = list(enumerate([f"switch {n}" for n in range(1, 7)], start=4))
        valid_switches = [
            switch_actions[i]
            for i, pokemon in enumerate(obs.request["side"]["pokemon"])
            if not pokemon["active"] and pokemon["condition"] != "0 fnt"
        ]
        if "wait" in obs.request:
            return []
        elif "forceSwitch" in obs.request:
            if "Revival Blessing" in obs.split_message:
                dead_switches = [
                    switch_actions[i]
                    for i, pokemon in enumerate(obs.request["side"]["pokemon"])
                    if not pokemon["active"] and pokemon["condition"] == "0 fnt"
                ]
                return dead_switches
            else:
                return valid_switches
        elif "active" in obs.request:
            move_switches = list(enumerate([f"move {n}" for n in range(1, 5)]))
            valid_moves = [
                move_switches[i]
                for i, move in enumerate(obs.request["active"][0]["moves"])
                if not ("disabled" in move and move["disabled"])
            ]
            if "trapped" in obs.request["active"][0] or "maybeTrapped" in obs.request["active"][0]:
                return valid_moves
            else:
                return valid_moves + valid_switches
        else:
            raise RuntimeError("Unknown request format encountered")
