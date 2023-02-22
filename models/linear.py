from typing import List
from torch import nn
import torch


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        return x


class LinearNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_block_out: int,
        feature_multipliers: List[float],
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_block = nn.Linear(in_features, input_block_out)
        self.feature_list = [input_block_out] + [
            int(input_block_out * i) for i in feature_multipliers
        ]
        self.middle_blocks = nn.Sequential()
        for i in range(len(self.feature_list) - 1):
            self.middle_blocks.append(
                LinearBlock(self.feature_list[i], self.feature_list[i + 1])
            )
        self.dropout = nn.Dropout(dropout)
        self.output_block = nn.Linear(self.feature_list[-1], out_features)

    def forward(self, x: torch.Tensor):
        x = self.input_block(x)
        x = self.middle_blocks(x)
        x = self.dropout(x)
        x = self.output_block(x)
        return x
