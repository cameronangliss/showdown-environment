import asyncio
from datetime import datetime
import json
import logging

from battle import Battle
from player import Player


async def main():
    logging.basicConfig(
        level=logging.INFO, filename="debug.log", filemode="w", format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    with open("config.json") as f:
        config = json.load(f)
    player1 = Player(config["username1"], config["password1"], logger)
    player2 = Player(config["username2"], config["password2"], logger)
    battle = Battle(player1, player2, logger)
    await battle.setup()
    counter = 1
    while True:
        winner = await battle.run_episode()
        time = datetime.now().strftime("%H:%M:%S")
        print(f"{time}: {winner} wins game {counter}")
        counter += 1


if __name__ == "__main__":
    asyncio.run(main())
