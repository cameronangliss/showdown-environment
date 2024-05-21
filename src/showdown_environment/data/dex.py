import json

json_path = "src/showdown_environment/data/json"

pokedex = json.load(open(f"{json_path}/gen4pokedex.json"))
movedex = json.load(open(f"{json_path}/gen4moves.json"))
typedex = json.load(open(f"{json_path}/gen4typechart.json"))
abilitydex = json.load(open(f"{json_path}/abilities.json"))
itemdex = json.load(open(f"{json_path}/items.json"))
setdex = json.load(open(f"{json_path}/gen4randombattle.json"))
