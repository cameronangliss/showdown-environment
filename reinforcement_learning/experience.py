from typing import NamedTuple

from poke_dojo.state.battle import Battle


class Experience(NamedTuple):
    turn: int
    total_turns: int
    state: Battle
    action: int | None
    next_state: Battle
    reward: int
    done: bool
