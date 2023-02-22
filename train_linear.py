import pandas as pd
import numpy as np
import os
from scipy.io import wavfile
import torchaudio
from defs import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from featurizers.linear import LinearFeaturizer
from featurizers.spectogram import MelFeaturizer
from models.linear import LinearNet
from dataset import SimpleDataset
from pl_module import PlModule

N_MELS = 196
N_LINEAR_FEATURES = 8
BATCH_SIZE = 32
INPUT_BLOCK_OUT_FEATURES = 4096

df = pd.read_csv("data/ESC-50-master/meta/esc50.csv")
train_df = df.loc[df.fold != 1].reset_index(drop=True)
val_df = df.loc[df.fold == 1].reset_index(drop=True)


train_dataset = SimpleDataset(train_df, AUDIO_PATH)
val_dataset = SimpleDataset(val_df, AUDIO_PATH)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True
)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=3)

spectogram_featurizer = MelFeaturizer(n_mels=N_MELS)
linear_featurizer = LinearFeaturizer(spectogram_featurizer)
model = LinearNet(
    N_MELS * N_LINEAR_FEATURES, 50, INPUT_BLOCK_OUT_FEATURES, [2, 2, 2], 0
)
pl_module = PlModule(model, linear_featurizer, 0.0001)
logger = TensorBoardLogger("runs", name="Linear model")
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[3],
    logger=logger,
    check_val_every_n_epoch=20,
    max_epochs=10000,
)
trainer.fit(
    pl_module,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
