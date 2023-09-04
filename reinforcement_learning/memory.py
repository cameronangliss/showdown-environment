import random
from torch import Tensor
from typing import Deque, NamedTuple


class Experience(NamedTuple):
    state_tensor: Tensor
    action: int | None
    next_state_tensor: Tensor
    reward: int
    done: bool


class Memory(Deque[Experience]):
    def __init__(self, maxlen: int):
        super().__init__([], maxlen=maxlen)

    def sample(self, batch_size: int):
        return random.sample(self, batch_size)
