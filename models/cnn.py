from typing import List, Tuple, Type
from torch import nn
import torch


class CNN(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        imsize: Tuple[int, int],
        backbone: Type[nn.Module],
    ) -> None:
        """_summary_

        Args:
            in_features (_type_): _description_
            out_features (_type_): _description_
            imsize (Tuple[int, int]): _description_
            backbone (Type[nn.Module]): backbone must be from timm
        """
        super().__init__()
        self.imsize = imsize
        self.input_conv = nn.Conv2d(in_features, 3, 1)
        self.backbone = backbone
        self.output = nn.Linear(1000, out_features)

    def forward(self, x):
        x = nn.functional.interpolate(x, self.imsize)
        x = self.input_conv(x)
        x = self.backbone(x)
        x = self.output(x)
        return x
