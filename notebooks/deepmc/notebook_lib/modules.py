from typing import Tuple

import pytorch_lightning as pl
import torch
from notebook_lib.models import DeepMCModel, DeepMCPostModel
from torch import Tensor, nn


class DeepMCTrain(pl.LightningModule):
    def __init__(
        self,
        first_channels: int,
        rest_channels: int,
        first_encoder_channels: int,
        rest_encoder_channels: Tuple[int, int, int],
        sequence_length: int,
        kernel_size: int,
        num_inputs: int,
        encoder_layers: int = 2,
        encoder_models: int = 4,
        encoder_heads: int = 4,
        encoder_ff_features: int = 16,
        encoder_lr: float = 0.1,
        dropout: float = 0.2,
        batch_first: bool = True,
        return_sequence: bool = True,
    ):
        super().__init__()

        self.deepmc = DeepMCModel(
            first_channels=first_channels,
            rest_channels=rest_channels,
            first_encoder_channels=first_encoder_channels,
            rest_encoder_channels=rest_encoder_channels,
            sequence_length=sequence_length,
            kernel_size=kernel_size,
            num_inputs=num_inputs,
            encoder_layers=encoder_layers,
            encoder_features=encoder_models,
            encoder_heads=encoder_heads,
            encoder_ff_features=encoder_ff_features,
            encoder_dropout=encoder_lr,
            dropout=dropout,
            batch_first=batch_first,
            return_sequence=return_sequence,
        )

        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, x: Tensor):
        y = self.deepmc(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, eps=1e-07)
        return optimizer

    def training_step(self, train_batch: Tensor, _):
        x, y = train_batch[:6], train_batch[6]
        y_hat = self.deepmc(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss/total", loss)
        return loss

    def validation_step(self, validation_batch: Tensor, _):
        x, y = validation_batch[:6], validation_batch[6]
        y_hat = self.deepmc(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss/total", loss, on_epoch=True)
        return loss


class DeepMCPostTrain(pl.LightningModule):
    def __init__(
        self,
        first_in_features: int,
        first_out_features: int = 48,
        second_out_features: int = 96,
        out_features: int = 24,
    ):
        super().__init__()

        self.deepmc = DeepMCPostModel(
            first_in_features=first_in_features,
            first_out_features=first_out_features,
            second_out_features=second_out_features,
            out_features=out_features,
        )

        self.loss = nn.L1Loss(reduction="sum")

    def forward(self, x: Tensor):
        y = self.deepmc(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch: Tensor, _):
        x, y = batch
        y_hat = self.deepmc(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss/total", loss)
        return loss

    def validation_step(self, batch: Tensor, _):
        x, y = batch
        y_hat = self.deepmc(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss/total", loss, on_epoch=True)
        return loss
