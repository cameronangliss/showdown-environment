import asyncio
import json
import logging
import os
import requests
import re
import torch

from trainer import Trainer
from model import Model
from env import Env
from player import Player


def update_json_file(filename: str):
    response = requests.get(f"https://play.pokemonshowdown.com/data/{filename}.js")
    js_text = response.text
    i = js_text.index("{")
    js_literal = js_text[i:-1]
    json_text = re.sub(
        r'(?<![\w\d"])([a-zA-Z0-9_]+)(:)',
        r'"\1"\2',
        js_literal,
    )
    with open(f"json/{filename}.json", "w") as file:
        file.write(json_text)


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

    # update json files
    update_json_file("pokedex")
    update_json_file("moves")
    update_json_file("abilities")
    update_json_file("items")
    update_json_file("typechart")

    # construct and run trainer
    trainer = Trainer(model, env)
    await trainer.train(num_episodes, epsilon, gamma, alpha)

    # save progress
    torch.save(trainer.model.state_dict(), f"saves/{file_name}.pth")


if __name__ == "__main__":
    asyncio.run(main())
