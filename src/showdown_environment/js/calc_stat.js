const { calcStat } = require('@smogon/calc')

var input = ''
process.stdin.on('data', (chunk) => input += chunk)
process.stdin.on('end', () => {
    const data = JSON.parse(input)
    const result = calcStat(
        data['gen'],
        data['id'],
        data['base'],
        data['iv'],
        data['ev'],
        data['level'],
        data['nature'],
    );
    console.log(result)
})
