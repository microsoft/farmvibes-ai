from typing import Any, Dict

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        in_channels: int,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        classes: int = 1,
        num_epochs: int = 10,
    ):
        """Initialize a new Segmentation Model instance.

        Args:
            lr: learning rate.
            weight_decay: amount of weight decay regularization.
            in_channels: number of input channels of the network.
                Needs to match the number of bands/channels of the stacked NVDI raster.
            encoder_name: name of the encoder used for the Unet.
                See segmentation_models_pytorch for more information.
            encoder_weights: name of the pretrained weights for the encoder.
                Use 'imagenet' or None (random weights).
            classes: number of output classes.
                As we are doing a binary crop vs. non-crop segmentation, we use the default value.
            num_epochs: number of training epochs. Used for the cosine annealing scheduler.
        """
        super().__init__()
        self.save_hyperparameters()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = smp.FPN(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=in_channels,
            classes=self.classes,
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        metrics = torchmetrics.MetricCollection(
            {
                "ap": torchmetrics.BinnedAveragePrecision(num_classes=1, thresholds=100),
                "acc": torchmetrics.Accuracy(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]

    def _shared_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        pred = self(batch["image"])
        for t in pred, batch["mask"]:
            assert torch.all(torch.isfinite(t))
        loss = self.loss(pred, batch["mask"])

        return {"loss": loss, "preds": pred.detach(), "target": batch["mask"]}

    def _shared_step_end(
        self, outputs: Dict[str, Any], metrics: torchmetrics.MetricCollection, prefix: str
    ) -> None:
        m = metrics(outputs["preds"].sigmoid().flatten(), outputs["target"].flatten().to(torch.int))
        self.log(f"{prefix}_loss", outputs["loss"])
        self.log_dict(m)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._shared_step(batch, batch_idx)

    def training_step_end(self, outputs: Dict[str, Any]) -> None:
        self._shared_step_end(outputs, self.train_metrics, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._shared_step(batch, batch_idx)

    def validation_step_end(self, outputs: Dict[str, Any]) -> None:
        return self._shared_step_end(outputs, self.val_metrics, "val")
