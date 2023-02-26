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
import timm

from featurizers.linear import LinearFeaturizer
from featurizers.spectogram import MelFeaturizer, MFCCFeaturizer
from models.cnn import CNN
from dataset import SimpleDataset, MixUpDataset
from pl_module import PlModule

N_MELS = 224
N_LINEAR_FEATURES = 8
BATCH_SIZE = 32
VAL_FOLD = 5
USE_MIXUP = False
MIXUP_RATIO = 0.5
ALPHA = 0.6
TRANSFORMS = TRANSFORM_SET_1

df = pd.read_csv("data/ESC-50-master/meta/esc50.csv")
train_df = df.loc[df.fold != VAL_FOLD].reset_index(drop=True)
val_df = df.loc[df.fold == VAL_FOLD].reset_index(drop=True)

if USE_MIXUP:
    train_dataset = MixUpDataset(train_df, AUDIO_PATH, TRANSFORMS, MIXUP_RATIO, ALPHA)
    val_dataset = MixUpDataset(val_df, AUDIO_PATH, None, 0, 0)
    loss_fn = torch.nn.BCEWithLogitsLoss()
else:
    train_dataset = SimpleDataset(train_df, AUDIO_PATH, TRANSFORMS)
    val_dataset = SimpleDataset(val_df, AUDIO_PATH, None)
    loss_fn = torch.nn.CrossEntropyLoss()
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=50,
    shuffle=True,
    persistent_workers=True,
)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=5)

spectogram_featurizer = MelFeaturizer(n_mels=N_MELS)
backbone = timm.create_model("efficientnet_b0", pretrained=True)
model = CNN(1, 50, (224, 224), backbone)
pl_module = PlModule(model, spectogram_featurizer, loss_fn, 0.0001)
logger = TensorBoardLogger(
    "runs", name="CNN model", version="efficientnet_b0 augset1 fold5"
)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[3],
    logger=logger,
    check_val_every_n_epoch=20,
    max_epochs=10000,
    enable_progress_bar=True,
    num_sanity_val_steps=0,
)
trainer.fit(
    pl_module,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
