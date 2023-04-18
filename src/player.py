from __future__ import annotations

import asyncio
import json
from logging import Logger
import requests
import websockets


class Player:
    def __init__(self, username: str, password: str, logger: Logger):
        self.username = username
        self.password = password
        self.room = None
        self.request = None
        self.observations = None
        self.websocket = None
        self.logger = logger

    async def connect(self):
        while True:
            try:
                self.websocket = await websockets.connect("wss://sim3.psim.us/showdown/websocket")
                break
            except TimeoutError:
                self.logger.error("Connection attempt failed, retrying now")

    async def send_message(self, message: str):
        room = self.room or ""
        message = f"{room}|{message}"
        self.logger.info(f"\n{self.username.upper()} -> SERVER:\n{message}")
        await self.websocket.send(message)

    async def receive_message(self) -> str | None:
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
            self.logger.info(f"\nSERVER -> {self.username.upper()}:\n{response}")
            return response
        except asyncio.TimeoutError:
            self.logger.warning("receive_message timed out")
            return None

    async def find_message(self, message_type: str) -> list[str] | None:
        while True:
            message = await self.receive_message()
            if not message:
                return None
            split_message = message.split("|")
            match message_type:
                case "login":
                    if split_message[1] == "challstr":
                        return split_message
                case "games":
                    if split_message[1] == "updatesearch":
                        return split_message
                case "challenge":
                    if split_message[1] == "popup":
                        # too many games or challenges in too short a period of time cause a popup
                        raise RuntimeError(split_message[2])
                    elif (
                        split_message[1] == "pm"
                        and split_message[2] == f" {self.username}"
                        and "wants to battle!" in split_message[4]
                    ):
                        return split_message
                case "accept":
                    if (
                        split_message[1] == "pm"
                        and split_message[3] == f" {self.username}"
                        and "wants to battle!" in split_message[4]
                    ):
                        return split_message
                case "request":
                    if (
                        "win" in split_message
                        or "tie" in split_message
                        or (split_message[1] == "request" and split_message[2])
                    ):
                        return split_message
                case "observe":
                    if "t:" in split_message:
                        return split_message
                case "leave":
                    if self.room in split_message[0] and split_message[1] == "deinit":
                        return split_message

    async def login(self):
        split_message = await self.find_message("login")
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
        await self.find_message("games")
        split_message = await self.find_message("games")
        # If the second games message isn't found, then there are no games to forfeit
        if not split_message:
            return
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

    async def setup(self):
        self.room = None
        await self.connect()
        await self.login()
        await self.forfeit_games()

    async def challenge(self, opponent: Player, battle_format: str = "gen9randombattle", team: str | None = None):
        await self.send_message(f"/utm {team}")
        await self.send_message(f"/challenge {opponent.username}, {battle_format}")
        await self.find_message("challenge")

    async def accept(self, opponent: Player, team: str | None = None) -> str:
        await self.find_message("accept")
        await self.send_message(f"/utm {team}")
        await self.send_message(f"/accept {opponent.username}")
        # The first games message is always empty, so this is here to pass by that message.
        await self.find_message("games")
        split_message = await self.find_message("games")
        games = json.loads(split_message[2])["games"]
        room = list(games.keys())[0]
        return room

    async def join(self, room: str):
        await self.send_message(f"/join {room}")
        self.room = room

    async def timer_on(self):
        await self.send_message("/timer on")

    async def observe(self) -> tuple[bool, str]:
        split_message = await self.find_message("request")
        if "win" in split_message:
            i = split_message.index("win")
            return True, split_message[i + 1].strip()
        elif "tie" in split_message:
            return True, None
        else:
            self.request = json.loads(split_message[2])
            self.observations = await self.find_message("observe")
            return False, None

    async def choose(self, choice: str):
        rqid = self.request["rqid"]
        await self.send_message(f"/choose {choice}|{rqid}")

    async def leave(self):
        await self.send_message(f"/leave {self.room}")
        # gets rid of all messages having to do with the room being left
        await self.find_message("leave")
        self.room = None
        self.request = None
        self.observations = None

    async def logout(self):
        await self.send_message("/logout")
