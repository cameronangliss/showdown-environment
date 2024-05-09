import asyncio
import json
import os
from copy import deepcopy

import torch
from model import Model
from player import Player

from showdown_environment.showdown.environment import Environment


async def train():
    # load configuration
    with open("config.json") as f:
        config = json.load(f)
    # construct player
    alpha = float(config["alpha"])
    epsilon = float(config["epsilon"])
    gamma = float(config["gamma"])
    memory_length = int(config["memory_length"])
    hidden_layer_sizes = json.loads(config["hidden_layer_sizes"])
    model = Model(alpha, epsilon, gamma, memory_length, hidden_layer_sizes)
    player = Player(config["player_username"], config["player_password"], model)
    # load saved model with the same settings as `model` if one exists
    file_name = f"{alpha:.0e}_{epsilon}_{gamma}_{memory_length:.0e}_{hidden_layer_sizes}"
    if not os.path.exists("saves"):
        os.makedirs("saves")
    model_version = 0
    while os.path.exists(f"saves/mk{model_version + 1}_{file_name}.pt"):
        model_version += 1
    if model_version > 0:
        model.load_state_dict(torch.load(f"saves/mk{model_version}_{file_name}.pt"))  # type: ignore
        print(f"mk{model_version}_{file_name}.pt has been loaded.")
    else:
        torch.save(player.model.state_dict(), f"saves/mk{model_version}_{file_name}.pt")  # type: ignore
        print(f"mk{model_version}_{file_name}.pt has been initialized.")
    # train model
    while True:
        env_player = Player(
            config["env_player_username"], config["env_player_password"], deepcopy(player.model)
        )
        env = Environment(env_player)
        await player.improve(env, num_episodes=100, min_win_rate=0.55)
        model_version += 1
        print(f"Model has been upgraded to mk{model_version}!")
        torch.save(player.model.state_dict(), f"saves/mk{model_version}_{file_name}.pt")  # type: ignore


if __name__ == "__main__":
    asyncio.run(train())
