from enum import Enum, auto
from logging import Logger
from typing import Any

import websockets.client as ws


class PopupError(Exception):
    def __init__(self, *args: Any):
        super().__init__(*args)


class MessageType(Enum):
    LOGIN = auto()
    GAMES = auto()
    CHALLENGE = auto()
    CANCEL = auto()
    ACCEPT = auto()
    OBSERVE = auto()
    LEAVE = auto()


class Client:
    __websocket: ws.WebSocketClientProtocol | None
    __username: str
    __logger: Logger

    def __init__(self, username: str, logger: Logger):
        self.__websocket = None
        self.__username = username
        self.__logger = logger

    async def connect(self):
        while True:
            try:
                self.__websocket = await ws.connect("wss://sim3.psim.us/showdown/websocket")
                break
            except TimeoutError:
                self.__logger.error("Connection attempt failed, retrying now")

    async def send_message(self, room: str | None, message: str):
        room_str = room or ""
        message = f"{room_str}|{message}"
        self.__logger.info(message)
        if self.__websocket:
            await self.__websocket.send(message)
        else:
            raise ConnectionError("Cannot send message without established websocket")

    async def receive_message(self) -> str:
        if self.__websocket:
            response = str(await self.__websocket.recv())
        else:
            raise ConnectionError("Cannot receive message without established websocket")
        self.__logger.info(response)
        return response

    async def find_message(self, room: str | None, message_type: MessageType) -> list[str]:
        while True:
            message = await self.receive_message()
            split_message = message.split("|")
            match message_type:
                case MessageType.LOGIN:
                    if split_message[1] == "challstr":
                        return split_message
                case MessageType.GAMES:
                    if split_message[1] == "popup":
                        # Popups encountered when searching for games message in the past:
                        # 1. Due to high load, you are limited to 12 battles and team validations every 3 minutes.
                        # NOTE: This popup occurs in response to the player accepting a challenge, but manifests when looking for
                        # the games message.
                        raise PopupError(split_message[2])
                    elif split_message[1] == "updatesearch":
                        return split_message
                case MessageType.CHALLENGE:
                    if split_message[1] == "popup":
                        # Popups encountered when searching for challenge message in the past:
                        # 1. Due to high load, you are limited to 12 battles and team validations every 3 minutes.
                        # 2. You challenged less than 10 seconds after your last challenge! It's cancelled in case it's a misclick.
                        # 3. You are already challenging someone. Cancel that challenge before challenging someone else.
                        # 4. The server is restarting. Battles will be available again in a few minutes.
                        raise PopupError(split_message[2])
                    elif (
                        split_message[1] == "pm"
                        and split_message[2] == f" {self.__username}"
                        and "wants to battle!" in split_message[4]
                    ):
                        return split_message
                case MessageType.CANCEL:
                    if split_message[1] == "popup":
                        # Popups encountered when searching for cancel message in the past:
                        # 1. You are not challenging <opponent_username>. Maybe they accepted/rejected before you cancelled?
                        raise PopupError(split_message[2])
                    elif (
                        split_message[1] == "pm"
                        and split_message[2] == f" {self.__username}"
                        and "cancelled the challenge." in split_message[4]
                    ):
                        return split_message
                case MessageType.ACCEPT:
                    if (
                        split_message[1] == "pm"
                        and split_message[3] == f" {self.__username}"
                        and "wants to battle!" in split_message[4]
                    ):
                        return split_message
                case MessageType.OBSERVE:
                    is_request = split_message[1] == "request" and split_message[2]
                    is_protocol = "\n" in split_message
                    if is_request or is_protocol:
                        return split_message
                case MessageType.LEAVE:
                    if room is not None and room in split_message[0] and split_message[1] == "deinit":
                        return split_message
