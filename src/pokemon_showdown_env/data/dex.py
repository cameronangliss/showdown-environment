import json
from importlib.resources import files
from typing import Any

pokedex: dict[str, dict[str, Any]] = {}
movedex: dict[str, dict[str, Any]] = {}
typedex: dict[str, dict[str, Any]] = {}
abilitydex: dict[str, dict[str, Any]] = {}
itemdex: dict[str, dict[str, Any]] = {}

for i in range(1, 10):
    pokedex[f"gen{i}"] = json.loads(files("src/pokemon_showdown_env").joinpath(f"data/json/gen{i}", "pokedex.json").read_text())
    movedex[f"gen{i}"] = json.loads(files("src/pokemon_showdown_env").joinpath(f"data/json/gen{i}", "movedex.json").read_text())
    typedex[f"gen{i}"] = json.loads(files("src/pokemon_showdown_env").joinpath(f"data/json/gen{i}", "typedex.json").read_text())
    abilitydex[f"gen{i}"] = json.loads(files("src/pokemon_showdown_env").joinpath(f"data/json/gen{i}", "abilitydex.json").read_text())
    itemdex[f"gen{i}"] = json.loads(files("src/pokemon_showdown_env").joinpath(f"data/json/gen{i}", "itemdex.json").read_text())
