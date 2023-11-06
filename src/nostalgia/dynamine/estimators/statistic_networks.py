import torch
from torch import nn
from mutual_information.configs.mine_config import MineConfig


class StatisticsNetwork(nn.Module):
    """
    """
    def __init__(self, config: MineConfig):
        super().__init__()
        x_dim = config.data.dimension
        y_dim = config.data.dimension
        self.statistics_network = nn.Sequential(
            nn.Linear(x_dim + y_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, X, Y, dim=1):
        XY = torch.cat([X, Y], dim=1)
        T = self.statistics_network(XY)
        return T