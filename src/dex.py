import json
import subprocess
from typing import Any

subprocess.call(["node", "dex.js"])

pokedex: dict[str, Any] = {}
movedex: dict[str, Any] = {}
typedex: dict[str, Any] = {}
abilitydex: dict[str, Any] = {}
itemdex: dict[str, Any] = {}
for i in range(1, 10):
    pokedex[f"gen{i}"] = json.load(open(f"json/gen{i}/pokedex.json"))
    movedex[f"gen{i}"] = json.load(open(f"json/gen{i}/movedex.json"))
    typedex[f"gen{i}"] = json.load(open(f"json/gen{i}/typedex.json"))
    abilitydex[f"gen{i}"] = json.load(open(f"json/gen{i}/abilitydex.json"))
    itemdex[f"gen{i}"] = json.load(open(f"json/gen{i}/itemdex.json"))
