# showdown-environment

## Running example code

1. Clone the repo.
1. Open a python virtual environment if necessary, and run `pip install -r requirements.txt` to install dependencies.
1. Run `cp config_example.json config.json`.
1. Fill in `config.json`. Inspect code in `example` folder to see how `config.json` is used.
1. Run `python3 example/main.py` to see the example code in action.

**Disclaimer: Functionality has only been tested in Ubuntu 20.04.

## Update data stored in `showdown_environment/data/json` folder

1. Ensure you have Node and npm installed.
1. Run the following code:
    ```
    cd scripts
    npm i
    cd ..
    node scripts/update_data.js
    ```
