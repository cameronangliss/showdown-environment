import random
from typing import Deque, NamedTuple

from torch import Tensor


class Experience(NamedTuple):
    state_tensor: Tensor
    action: int | None
    next_state_tensor: Tensor
    reward: int
    done: bool


class Memory(Deque[Experience]):
    def __init__(self, experiences: list[Experience], maxlen: int | None = None):
        super().__init__(experiences, maxlen=maxlen)

    def sample(self, batch_size: int):
        return random.sample(self, batch_size)
