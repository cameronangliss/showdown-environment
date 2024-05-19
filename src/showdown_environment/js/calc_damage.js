const { calculate, Pokemon, Move } = require('@smogon/calc')

var input = ''
process.stdin.on('data', (chunk) => input += chunk)
process.stdin.on('end', () => {
    const data = JSON.parse(input)
    const result = calculate(
        data['gen'],
        new Pokemon(data['gen'], data['attacker'], data['attacker_info']),
        new Pokemon(data['gen'], data['defender'], data['defender_info']),
        new Move(data['gen'], data['move']),
    );
    console.log(result.damage)
})
