from typing import List, Union

import pytorch_lightning as pl
from notebook_lib.base_model import BatchTGCN
from torch import Tensor, nn
from torch.optim import Adagrad

from .schema import BatchTGCNInputs


class BatchTGCNTrain(pl.LightningModule):
    def __init__(
        self,
        inputs: BatchTGCNInputs,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.gnn = BatchTGCN(inputs)
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, batch: Union[Tensor, List[Tensor]]):
        y_hat, _, _ = self.gnn(batch)
        return y_hat

    def configure_optimizers(self):
        optimizer = Adagrad(
            self.parameters(),
            lr=self.learning_rate,
            initial_accumulator_value=1e-6,
            eps=1e-6,
            weight_decay=1e-6,
        )
        return optimizer

    def training_step(self, train_batch: Union[Tensor, List[Tensor]], _):
        _, _, _, node_labels = self.gnn.get_batch(train_batch, self.gnn.use_edge_weights)
        y = node_labels
        y_hat, _, _ = self.gnn(train_batch)
        loss = self.loss(y_hat, y.reshape(y_hat.shape))
        self.log("train_loss/total", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, validation_batch: Union[Tensor, List[Tensor]], _):
        _, _, _, node_labels = self.gnn.get_batch(validation_batch, self.gnn.use_edge_weights)
        y = node_labels
        y_hat, _, _ = self.gnn(validation_batch)
        loss = self.loss(y_hat, y.reshape(y_hat.shape))
        self.log("val_loss/total", loss, on_epoch=True, prog_bar=True)
        return loss
