import json
from typing import Any

pokedex: dict[str, dict[str, Any]] = {}
movedex: dict[str, dict[str, Any]] = {}
typedex: dict[str, dict[str, Any]] = {}
abilitydex: dict[str, dict[str, Any]] = {}
itemdex: dict[str, dict[str, Any]] = {}

json_path = "showdown_environment/data/json"

for i in range(1, 10):
    pokedex[f"gen{i}"] = json.load(open(f"{json_path}/gen{i}/pokedex.json"))
    movedex[f"gen{i}"] = json.load(open(f"{json_path}/gen{i}/movedex.json"))
    typedex[f"gen{i}"] = json.load(open(f"{json_path}/gen{i}/typedex.json"))
    abilitydex[f"gen{i}"] = json.load(open(f"{json_path}/gen{i}/abilitydex.json"))
    itemdex[f"gen{i}"] = json.load(open(f"{json_path}/gen{i}/itemdex.json"))
