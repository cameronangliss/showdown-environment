import asyncio
import json
import logging
import os
import random
from datetime import datetime

import torch

from env import Env
from model import Model
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

    # construct model
    alpha = float(config["alpha"])
    epsilon = float(config["epsilon"])
    gamma = float(config["gamma"])
    hidden_dims = json.loads(config["hidden_dims"])
    model = Model(env, alpha, epsilon, gamma, hidden_dims)

    # load saved model with the same settings as `model` if one exists
    file_name = f"{alpha}_{epsilon}_{gamma}_{hidden_dims}"
    if os.path.exists(f"saves/{file_name}.pt"):
        model.load_state_dict(torch.load(f"saves/{file_name}.pt"))  # type: ignore

    # train model
    num_episodes = int(config["num_episodes"])
    await model.env.setup()
    random_formats = [f"gen{n}randombattle" for n in range(4, 10)]
    for i in range(num_episodes):
        winner = await model.run_episode(random.choice(random_formats))
        time = datetime.now().strftime("%H:%M:%S")
        print(f"{time}: {winner} wins game {i + 1}")
    await model.env.close()

    # save progress
    if not os.path.exists("saves"):
        os.makedirs("saves")
    torch.save(model.state_dict(), f"saves/{file_name}.pt")  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
