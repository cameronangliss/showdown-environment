import asyncio
import json
import logging

from trainer import Trainer
from model import Model
from env import Env
from player import Player


async def main():
    with open("config.json") as f:
        config = json.load(f)
    logging.basicConfig(
        level=logging.INFO,
        filename="debug.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s()\n%(message)s\n",
    )
    logger = logging.getLogger()
    player1 = Player(config["username1"], config["password1"], logger)
    player2 = Player(config["username2"], config["password2"], logger)
    model = Model(10, 20, 10)
    env = Env(player1, player2, logger)
    trainer = Trainer(model, env)
    await trainer.train(num_episodes=100, epsilon=0.1, gamma=0.9, alpha=0.01)


if __name__ == "__main__":
    asyncio.run(main())
