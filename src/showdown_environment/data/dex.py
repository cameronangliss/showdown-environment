import json

json_path = "src/showdown_environment/data/json"

pokedex = json.load(open(f"{json_path}/pokedex.json"))
movedex = json.load(open(f"{json_path}/movedex.json"))
typedex = json.load(open(f"{json_path}/typedex.json"))
abilitydex = json.load(open(f"{json_path}/abilitydex.json"))
itemdex = json.load(open(f"{json_path}/itemdex.json"))
