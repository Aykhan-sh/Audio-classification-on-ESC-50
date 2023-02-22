import torch
from torch import nn


class LinearFeaturizer(nn.Module):
    """
    extracts next features by frequencies:
    mean
    std
    skew
    kurtosis
    25th quantile
    50th quantile
    75th quantile
    maximums
    """

    def __init__(self, spectogram_featurizer, nan_to_num_value: float = -1) -> None:
        super().__init__()
        self.spectogram_featurizer = spectogram_featurizer
        self.nan_to_num_value = nan_to_num_value

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.shape[0]
        spectogram = self.spectogram_featurizer(batch)
        spectogram = spectogram.mean(1)
        spectogram = spectogram.permute(2, 0, 1)

        means = torch.mean(spectogram, dim=0)
        diffs = spectogram - means
        var = torch.mean(torch.pow(diffs, 2.0), dim=0)
        stds = torch.pow(var, 0.5)
        zscores = diffs / stds
        skews = torch.mean(torch.pow(zscores, 3.0), dim=0)
        kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=0) - 3.0

        q25 = spectogram.quantile(0.25, dim=0)
        q50 = spectogram.quantile(0.5, dim=0)
        q75 = spectogram.quantile(0.75, dim=0)
        maxes = spectogram.quantile(1, dim=0)

        features = torch.stack([means, stds, skews, kurtoses, q25, q50, q75, maxes])
        features = features.nan_to_num(self.nan_to_num_value)
        features = features.permute(1, 0, 2)
        return features.reshape(batch_size, -1)
