import json
import os
import re

import requests


def __scrape_js_file(filename: str):
    response = requests.get(f"https://play.pokemonshowdown.com/data/{filename}.js")
    js_text = response.text
    i = js_text.index("{")
    js_literal = js_text[i:-1]
    json_text = re.sub(
        r"(?<![\"\w\s\-])(\w+)(:)",
        r'"\1"\2',
        js_literal,
    )
    json_obj = json.loads(json_text)
    if not os.path.exists("json"):
        os.makedirs("json")
    with open(f"json/{filename}.json", "w") as f:
        json.dump(json_obj, f, indent=4)


def __scrape_ts_file(filename: str):
    response = requests.get(f"https://play.pokemonshowdown.com/data/pokemon-showdown/data/{filename}.ts")
    ts_text = response.text
    i = ts_text.find("=")
    j = ts_text.find("{", i)
    ts_literal = re.sub(r",\s*}", "}", ts_text[j:]).replace("'", '"')
    json_text = re.sub(
        r"(?<![\"\w])(\w+)(:)",
        r'"\1"\2',
        ts_literal,
    )
    decoder = json.JSONDecoder(strict=False)
    json_obj, _ = decoder.raw_decode(json_text)
    if not os.path.exists("json"):
        os.makedirs("json")
    with open(f"json/{filename}.json", "w") as f:
        json.dump(json_obj, f, indent=4)


__scrape_js_file("pokedex")
__scrape_js_file("moves")
__scrape_js_file("typechart")
__scrape_ts_file("natures")
__scrape_js_file("abilities")
__scrape_js_file("items")


pokedex = json.load(open("json/pokedex.json"))
movedex = json.load(open("json/moves.json"))
typedex = json.load(open("json/typechart.json"))
naturedex = json.load(open("json/natures.json"))
abilitydex = json.load(open("json/abilities.json"))
itemdex = json.load(open("json/items.json"))
