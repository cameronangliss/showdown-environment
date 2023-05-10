import asyncio
import json
import logging
import os

import torch

from env import Env
from model import Model
from player import Player
from trainer import Trainer


async def main():
    # load configuration
    with open("config.json") as f:
        config = json.load(f)

    # construct training environment
    logging.basicConfig(
        level=logging.INFO,
        filename="debug.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s()\n%(message)s\n",
    )
    logger = logging.getLogger()
    player1 = Player(config["username1"], config["password1"], logger)
    player2 = Player(config["username2"], config["password2"], logger)
    env = Env(player1, player2, logger)

    # construct model or load save file if one exists
    alpha = float(config["alpha"])
    epsilon = float(config["epsilon"])
    gamma = float(config["gamma"])
    hidden_dims = json.loads(config["hidden_dims"])
    model = Model(alpha, epsilon, gamma, hidden_dims)
    file_name = f"{alpha}_{epsilon}_{gamma}_{hidden_dims}"
    if os.path.exists(f"saves/{file_name}.pt"):
        model.load_state_dict(load(f"saves/{file_name}.pt"))  # type: ignore

    # construct and run trainer
    num_episodes = int(config["num_episodes"])
    trainer = Trainer(model, env)
    await trainer.train(num_episodes)

    # save progress
    if not os.path.exists("saves"):
        os.makedirs("saves")
    torch.save(trainer.model.state_dict(), f"saves/{file_name}.pt")  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
