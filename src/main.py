import asyncio
import json
import logging
import os
from copy import deepcopy

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
    player = Player(config["username"], config["password"], logger)
    alt_player = Player(config["alt_username"], config["alt_password"], logger)
    env = Env(player, alt_player, logger)

    # construct model
    alpha = float(config["alpha"])
    epsilon = float(config["epsilon"])
    gamma = float(config["gamma"])
    hidden_dims = json.loads(config["hidden_dims"])
    model = Model(alpha, epsilon, gamma, hidden_dims)
    alt_model = Model(alpha, epsilon, gamma, hidden_dims)

    # load saved model with the same settings as `model` if one exists
    file_name = f"{alpha}_{epsilon}_{gamma}_{hidden_dims}"
    if not os.path.exists("saves"):
        os.makedirs("saves")
    elif os.path.exists(f"saves/{file_name}.pt") and os.path.exists(f"saves/{file_name}_current.pt"):
        model.load_state_dict(torch.load(f"saves/{file_name}_current.pt"))  # type: ignore
        alt_model.load_state_dict(torch.load(f"saves/{file_name}.pt"))  # type: ignore
    else:
        alt_model = deepcopy(model)
        torch.save(model.state_dict(), f"saves/{file_name}.pt")  # type: ignore
        torch.save(model.state_dict(), f"saves/{file_name}_current.pt")  # type: ignore

    # train model
    train_episodes = int(config["train_episodes"])
    eval_episodes = int(config["eval_episodes"])
    improve_attempts = int(config["improve_attempts"])
    await env.setup()
    for _ in range(improve_attempts):
        win_rate = await model.attempt_improve(alt_model, env, train_episodes, eval_episodes)
        torch.save(model.state_dict(), f"saves/{file_name}_current.pt")  # type: ignore
        if win_rate < 0.55:
            print("Improvement failed. Trying again now.")
        else:
            print("Improvement succeeded! Overwriting model now.")
            torch.save(model.state_dict(), f"saves/{file_name}.pt")  # type: ignore
            alt_model = deepcopy(model)
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
