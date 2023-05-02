import torch
import torch.nn as nn

from player import Observation


class Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def process_observation(self, obs: Observation) -> torch.Tensor:
        return torch.rand(self.input_dim)

    def forward(self, obs: Observation) -> torch.Tensor:
        x = self.process_observation(obs)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
