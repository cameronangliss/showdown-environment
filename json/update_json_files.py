import re
import requests


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


if __name__ == "__main__":
    update_json_file("pokedex")
    update_json_file("moves")
    update_json_file("abilities")
    update_json_file("items")
    update_json_file("typechart")
