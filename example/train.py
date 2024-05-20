import asyncio
import json
import os
from copy import deepcopy

import torch
from actor import Actor
from critic import Critic
from player import Player

from showdown_environment.showdown.environment import Environment


async def train():
    # construct player
    with open("config.json") as f:
        config = json.load(f)
    alpha = float(config["alpha"])
    epsilon = float(config["epsilon"])
    gamma = float(config["gamma"])
    memory_length = int(config["memory_length"])
    hidden_layer_sizes = json.loads(config["hidden_layer_sizes"])
    actor = Actor(alpha, epsilon, gamma, hidden_layer_sizes)
    critic = Critic(alpha, epsilon, gamma, hidden_layer_sizes)
    player = Player(
        config["player_username"],
        config["player_password"],
        actor,
        critic,
        memory_length=memory_length,
    )
    # load saved actor/critic with the same settings if one exists
    file_name = f"{alpha:.0e}_{epsilon}_{gamma}_{memory_length:.0e}_{hidden_layer_sizes}"
    if not os.path.exists("saves"):
        os.makedirs("saves")
    player_version = 0
    while os.path.exists(f"saves/actor_v{player_version + 1}_{file_name}.pt"):
        player_version += 1
    if player_version > 0:
        player.actor.load_state_dict(torch.load(f"saves/actor_v{player_version}_{file_name}.pt"))  # type: ignore
        player.critic.load_state_dict(torch.load(f"saves/critic_v{player_version}_{file_name}.pt"))  # type: ignore
        print(f"{player.username} v{player_version} has been loaded.")
    else:
        torch.save(player.actor.state_dict(), f"saves/actor_v{player_version}_{file_name}.pt")  # type: ignore
        torch.save(player.critic.state_dict(), f"saves/critic_v{player_version}_{file_name}.pt")  # type: ignore
        print(f"{player.username} v{player_version} has been initialized.")
    # train actor
    while True:
        env_player = Player(
            config["env_player_username"],
            config["env_player_password"],
            deepcopy(player.actor),
            deepcopy(player.critic),
        )
        env = Environment(env_player)
        await player.improve(env, num_episodes=100, min_win_rate=0.55)
        player_version += 1
        print(f"{player.username} has been upgraded to v{player_version}!")
        torch.save(player.actor.state_dict(), f"saves/actor_v{player_version}_{file_name}.pt")  # type: ignore
        torch.save(player.critic.state_dict(), f"saves/critic_v{player_version}_{file_name}.pt")  # type: ignore


if __name__ == "__main__":
    asyncio.run(train())
