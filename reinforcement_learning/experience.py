from typing import NamedTuple

from torch import Tensor


class Experience(NamedTuple):
    state_tensor: Tensor
    action: int | None
    next_state_tensor: Tensor
    reward: int
    done: bool
