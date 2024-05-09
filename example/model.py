import torch
import torch.nn as nn
from memory import Experience, Memory
from torch import Tensor


class Model(nn.Module):
    __alpha: float
    epsilon: float
    __gamma: float
    memory_length: int
    __layers: nn.Sequential

    def __init__(
        self,
        alpha: float,
        epsilon: float,
        gamma: float,
        memory_length: int,
        hidden_layer_sizes: list[int],
    ):
        super().__init__()  # type: ignore
        self.__alpha = alpha
        self.epsilon = epsilon
        self.__gamma = gamma
        self.memory_length = memory_length
        self.memory = Memory([], maxlen=memory_length)
        layer_sizes = [1504, *hidden_layer_sizes, 26]
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()]
        self.__layers = nn.Sequential(*layers[:-1])
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.__alpha, momentum=0.9, weight_decay=1e-4
        )
        # Move the model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.__layers(x)

    def update(self, experience: Experience):
        if experience.done:
            q_target = torch.tensor(experience.reward)
        else:
            next_q_values = self.forward(experience.next_state_tensor)
            q_target = experience.reward + self.__gamma * torch.max(next_q_values)  # type: ignore
        q_values = self.forward(experience.state_tensor)
        q_estimate = q_values[experience.action]
        td_error = q_target - q_estimate
        loss = td_error**2
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        self.optimizer.step()
