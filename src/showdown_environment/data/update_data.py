import re
import requests
import os


def update_json_file(url: str, filename: str, file_ending: str):
    response = requests.get(f"{url}/{filename}{file_ending}")
    if file_ending == ".json":
        js_literal = response.text
    else:
        js_text = response.text
        i = js_text.index("{")
        js_literal = js_text[i:-1]
    json_text = re.sub(
        r"([{,])([a-zA-Z0-9_]+)(:)",
        r'\1"\2"\3',
        js_literal,
    )
    with open(f"src/showdown_environment/data/json/{filename}.json", "w") as file:
        file.write(json_text)


if __name__ == "__main__":
    if not os.path.exists("src/showdown_environment/data/json"):
        os.mkdir("src/showdown_environment/data/json")
    update_json_file("https://play.pokemonshowdown.com/data", "pokedex", ".js")
    update_json_file("https://play.pokemonshowdown.com/data", "moves", ".js")
    update_json_file("https://play.pokemonshowdown.com/data", "abilities", ".js")
    update_json_file("https://play.pokemonshowdown.com/data", "items", ".js")
    update_json_file("https://play.pokemonshowdown.com/data", "typechart", ".js")
    update_json_file("https://data.pkmn.cc/randbats", "gen4randombattle", ".json")
