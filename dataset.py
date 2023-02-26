import pandas as pd
import os
import torchaudio
from defs import *
from torch.utils.data import Dataset, DataLoader, Subset
import random
import torch
from typing import Optional


class SimpleDataset(Dataset):
    SR = 16_000

    def __init__(self, df: pd.DataFrame, audio_path: str, transforms):
        self.audio_path = audio_path
        self.df = df
        self.transforms = transforms

    def __getitem__(self, index: int):
        wav, label = self.get_one_item(index)
        if self.transforms is not None:
            wav = self.transforms(wav)
        return wav, label

    def get_one_item(self, index: int):
        path_to_wav = os.path.join(self.audio_path, self.df.filename[index])
        wav, sr = torchaudio.load(path_to_wav)
        label = self.df.target[index]
        return wav, label

    def __len__(self):
        return len(self.df)


class MixUpDataset(SimpleDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_path: str,
        transforms,
        mixup_ratio: float = 0.5,
        alpha: float = 0.5,
        num_labels: int = 50,
    ):
        super().__init__(df, audio_path, transforms)
        self.mixup_ratio = mixup_ratio
        self.alpha = alpha
        self.num_labels = num_labels

    def __getitem__(self, index: int):
        label = torch.zeros((self.num_labels), dtype=float)
        if random.random() < self.mixup_ratio:
            wav1, label_idx1 = self.get_one_item(index)
            wav2, label_idx2 = self.get_one_item(random.choice(range(len(self))))
            wav = self.alpha * wav1 + (1 - self.alpha) * wav2
            label[label_idx1] = self.alpha
            label[label_idx2] = 1 - self.alpha

        else:
            wav, label_idx = self.get_one_item(index)
            label[label_idx] = 1
        if self.transforms is not None:
            wav = self.transforms(wav)
        return wav, label
