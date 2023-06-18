const { Dex } = require('pokemon-showdown');
const fs = require('fs');

function transformData(jsonList) {
    result = {};
    for (let i = 0; i < jsonList.length; i++) {
        jsonObj = JSON.parse(JSON.stringify(jsonList[i]));
        if (jsonObj.id == "hiddenpower" && jsonObj.name != "Hidden Power") {
            splitName = jsonObj.name.split(" ");
            jsonObj.id += splitName[splitName.length - 1].toLowerCase();
        }
        result[jsonObj.id] = jsonObj;
    }
    return JSON.stringify(result, null, 4);
}

if (!fs.existsSync("json")) {
    fs.mkdirSync("json");
}

for (let i = 1; i <= 9; i++) {
    if (!fs.existsSync(`json/gen${i}`)) {
        fs.mkdirSync(`json/gen${i}`);
    }
    fs.writeFileSync(`json/gen${i}/pokedex.json`, transformData(Dex.mod(`gen${i}`).species.all()));
    fs.writeFileSync(`json/gen${i}/movedex.json`, transformData(Dex.mod(`gen${i}`).moves.all()));
    fs.writeFileSync(`json/gen${i}/typedex.json`, transformData(Dex.mod(`gen${i}`).types.all()));
    fs.writeFileSync(`json/gen${i}/abilitydex.json`, transformData(Dex.mod(`gen${i}`).abilities.all()));
    fs.writeFileSync(`json/gen${i}/itemdex.json`, transformData(Dex.mod(`gen${i}`).items.all()));
}
