import os
import re

import requests


def update_json_file(url: str, file: str):
    response = requests.get(f"{url}/{file}")
    if ".json" in file:
        json_text = response.text
    else:
        js_text = response.text
        i = js_text.index("{")
        js_literal = js_text[i:-1]
        json_text = re.sub(
            r"([{,])([a-zA-Z0-9_]+)(:)",
            r'\1"\2"\3',
            js_literal,
        )
        file += "on"
    with open(f"src/showdown_environment/data/json/{file}", "w") as f:
        f.write(json_text)


if __name__ == "__main__":
    if not os.path.exists("src/showdown_environment/data/json"):
        os.mkdir("src/showdown_environment/data/json")
    poke_env_url = "https://raw.githubusercontent.com/hsahovic/poke-env/master/src/poke_env"
    update_json_file(f"{poke_env_url}/data/static/pokedex", "gen4pokedex.json")
    update_json_file(f"{poke_env_url}/data/static/moves", "gen4moves.json")
    update_json_file("https://play.pokemonshowdown.com/data", "items.js")
    update_json_file("https://play.pokemonshowdown.com/data", "abilities.js")
    update_json_file(f"{poke_env_url}/data/static/typechart", "gen4typechart.json")
    update_json_file("https://data.pkmn.cc/randbats", "gen4randombattle.json")
