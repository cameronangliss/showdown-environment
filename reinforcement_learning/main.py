import asyncio
import json
import os

import torch
from model import Model


async def main():
    # load configuration
    with open("config.json") as f:
        config = json.load(f)

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
    improvements = int(config["num_improve"])
    for _ in range(improvements):
        await model.improve()
        torch.save(model.state_dict(), f"saves/{file_name}.pt")  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
