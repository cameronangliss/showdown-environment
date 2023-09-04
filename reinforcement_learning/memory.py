import random
from typing import Deque

from experience import Experience


class Memory:
    def __init__(self, capacity: int):
        experiences: list[Experience] = []
        self.memory = Deque(experiences, maxlen=capacity)

    def push(self, experiences: list[Experience]):
        for exp in experiences:
            self.memory.append(exp)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
