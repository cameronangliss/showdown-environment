from typing import Any, NamedTuple

import torch


class Experience(NamedTuple):
    turn: int
    total_turns: int
    state_tensor: torch.Tensor
    action: int | None
    next_state_tensor: torch.Tensor
    reward: int
    done: bool

    def to_json_serializable(self) -> list[Any]:
        return [
            self.turn,
            self.total_turns,
            [float(x.item()) for x in self.state_tensor],
            self.action,
            [float(x.item()) for x in self.next_state_tensor],
            self.reward,
            self.done,
        ]


def to_experience(json_list: list[Any]):
    return Experience(
        json_list[0],
        json_list[1],
        torch.tensor(json_list[2]).to("cuda" if torch.cuda.is_available() else "cpu"),
        json_list[3],
        torch.tensor(json_list[4]).to("cuda" if torch.cuda.is_available() else "cpu"),
        json_list[5],
        json_list[6],
    )
