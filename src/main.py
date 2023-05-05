import asyncio
import json
import logging
import os
import torch

from trainer import Trainer
from model import Model
from env import Env
from player import Player


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

    # construct model, load save file if one exists
    model_arch = json.loads(config["model_arch"])
    model = Model(*model_arch)
    epsilon = float(config["epsilon"])
    gamma = float(config["gamma"])
    alpha = float(config["alpha"])
    num_episodes = int(config["num_episodes"])
    file_name = f"{model_arch}_{epsilon}_{gamma}_{alpha}_{num_episodes}"
    if os.path.exists(f"saves/{file_name}.pth"):
        model.load_state_dict(torch.load(f"saves/{file_name}.pth"))

    # construct and run trainer
    trainer = Trainer(model, env)
    await trainer.train(num_episodes, epsilon, gamma, alpha)

    # save progress
    torch.save(trainer.model.state_dict(), f"saves/{file_name}.pth")


if __name__ == "__main__":
    asyncio.run(main())
