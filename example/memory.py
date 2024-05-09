import random
from typing import Deque

from showdown_environment.showdown.experience import Experience


class Memory(Deque[Experience]):
    def __init__(self, experiences: list[Experience], maxlen: int | None = None):
        super().__init__(experiences, maxlen=maxlen)

    def sample(self, batch_size: int):
        return random.sample(self, batch_size)
