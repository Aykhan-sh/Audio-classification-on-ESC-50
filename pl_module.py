import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from typing import Type
import torch


class PlModule(pl.LightningModule):
    def __init__(
        self,
        model: Type[nn.Module],
        featurizer: Type[nn.Module],
        loss_fn: Type[nn.Module],
        lr: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.featurizer = featurizer
        self.lr = lr
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        raw_audio, labels = batch
        features = self.featurizer(raw_audio)
        predictions = self.model(features)
        loss = self.loss_fn(predictions, labels)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        if (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch != 0:
            return
        losses = torch.tensor(0.0, device=self.device).float()
        for i in outputs:
            losses += i["loss"]
        losses = losses / len(outputs)
        self.log_dict({"loss/train": losses})

    def validation_step(self, batch, batch_idx):
        raw_audio, labels = batch
        features = self.featurizer(raw_audio)
        predictions = self.model(features)
        if len(labels.shape) > 1:
            accuracy = predictions.argmax(1) == labels.argmax(1)
        else:
            accuracy = predictions.argmax(1) == labels
        loss = self.loss_fn(predictions, labels)
        return {"corrects": accuracy, "loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        accuracy = []
        correct = torch.tensor(0.0, device=self.device)
        prediction_size = torch.tensor(0, device=self.device)
        losses = torch.tensor(0.0, device=self.device).float()
        for i in outputs:
            correct += i["corrects"].sum()
            prediction_size += i["corrects"].shape[0]
            losses += i["loss"]
        losses = losses / len(outputs)
        accuracy = correct / prediction_size
        self.log_dict({"accuracy/validation": accuracy, "loss/validation": losses})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
