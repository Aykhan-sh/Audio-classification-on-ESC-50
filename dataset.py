import pandas as pd
import os
import torchaudio
from defs import *
from torch.utils.data import Dataset, DataLoader, Subset


class SimpleDataset(Dataset):
    SR = 16_000

    def __init__(self, df: pd.DataFrame, audio_path: str):
        self.audio_path = audio_path
        self.df = df

    def __getitem__(self, index: int):
        path_to_wav = os.path.join(self.audio_path, self.df.filename[index])
        wav, sr = torchaudio.load(path_to_wav)
        label = self.df.target[index]
        return wav, label

    def __len__(self):
        return len(self.df)
