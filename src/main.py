import asyncio
import json
import logging
import os

import torch

from env import Env
from model import Experience, Model
from player import Player


async def main():
    # load configuration
    with open("config.json") as f:
        config = json.load(f)

    # construct training environment
    logging.basicConfig(
        level=logging.DEBUG,
        filename="debug.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s()\n%(message)s\n",
    )
    logger = logging.getLogger()
    player = Player(config["username"], config["password"], logger)
    alt_player = Player(config["alt_username"], config["alt_password"], logger)
    env = Env(player, alt_player, logger)

    # construct model
    alpha = float(config["alpha"])
    epsilon = float(config["epsilon"])
    gamma = float(config["gamma"])
    hidden_dims = json.loads(config["hidden_dims"])
    model = Model(alpha, epsilon, gamma, hidden_dims)

    # load saved model with the same settings as `model` if one exists
    file_name = f"{alpha}_{epsilon}_{gamma}_{hidden_dims}"
    if not os.path.exists("saves"):
        os.makedirs("saves")
    elif os.path.exists(f"saves/{file_name}.pt"):
        model.load_state_dict(torch.load(f"saves/{file_name}.pt"))  # type: ignore

    # train model
    iterations = int(config["iterations"])
    experiences: list[Experience] = []
    for _ in range(iterations):
        new_experiences, num_wins = await model.attempt_improve(env, experiences)
        if num_wins < 55:
            experiences += new_experiences
        else:
            experiences = []
            torch.save(model.state_dict(), f"saves/{file_name}.pt")  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
