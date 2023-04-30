import asyncio
from datetime import datetime
import json
import logging

from battle import Battle
from player import Player


async def main():
    with open("config.json") as f:
        config = json.load(f)
    logging.basicConfig(
        level=logging.INFO,
        filename="debug.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s\n%(message)s\n",
    )
    logger = logging.getLogger()
    player1 = Player(config["username1"], config["password1"], logger)
    player2 = Player(config["username2"], config["password2"], logger)
    battle = Battle(player1, player2, logger)
    await battle.setup()
    for i in range(10):
        winner = await battle.run_episode()
        time = datetime.now().strftime("%H:%M:%S")
        print(f"{time}: {winner} wins game {i + 1}")
    await battle.close()


if __name__ == "__main__":
    asyncio.run(main())
