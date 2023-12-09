# showdown-environment

## Description

This is a Python client-side API for interacting with the [Pokemon Showdown](https://pokemonshowdown.com) website. It also contains an environment to train AI agents.

This repository essentially handles 3 major obstacles in using Pokemon Showdown to train AI agents:
1. Connecting to the Pokemon Showdown server to send challenges, choose moves in battle, etc (see `showdown_environment/showdown/`)
1. Interpreting the protocol and request messages from Pokemon Showdown into turn-by-turn state objects, which AI agents can use to make informed decisions (see `showdown_environment/state/`)
1. Collecting the Pokedex, Movedex, Itemdex, Abilitydex, and Typedex from Pokemon Showdown, which can be used to interpret elements of the state objects (see `showdown_environment/data/`)

Please feel free to use/modify the code for your specific needs.

**Disclaimer: Functionality has only been tested in Ubuntu 20.04.

## Running example code

1. Clone the repo.
1. Open a Python virtual environment if necessary, and run `pip install -r requirements.txt` to install dependencies.
1. Run `cp config_example.json config.json`.
1. Fill in `config.json`. Inspect code in `example/` to see how `config.json` is used.
1. Run `python3 example/main.py` to see the example code in action.

## Update data stored in `showdown_environment/data/json/`

1. Ensure you have `node` and `npm` installed.
1. Run the following code:
    ```
    cd scripts
    npm i
    cd ..
    node scripts/update_data.js
    ```
