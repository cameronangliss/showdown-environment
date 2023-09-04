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
    hidden_layer_sizes = json.loads(config["hidden_layer_sizes"])
    model = Model(alpha, epsilon, gamma, hidden_layer_sizes)
    # load saved model with the same settings as `model` if one exists
    file_name = f"{alpha}_{epsilon}_{gamma}"
    if not os.path.exists("saves"):
        os.makedirs("saves")
    model_version = 0
    while os.path.exists(f"saves/mk{model_version + 1}_{file_name}.pt"):
        model_version += 1
    if model_version > 0:
        model.load_state_dict(torch.load(f"saves/mk{model_version}_{file_name}.pt"))  # type: ignore
        print("Saved model has successfully loaded.")
    else:
        print("New model initialized.")
    # train model
    num_improve = int(config["num_improve"])
    for _ in range(num_improve):
        await model.improve()
        model_version += 1
        print(f"Model has been upgraded to mk{model_version}!")
        torch.save(model.state_dict(), f"saves/mk{model_version}_{file_name}.pt")  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
