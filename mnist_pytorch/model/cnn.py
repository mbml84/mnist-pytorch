from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class CNN(LightningModule):

    MODEL_NAME = 'mnist-cnn'

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.example_input_array = torch.Tensor(32, 1, 28, 28)

    def _log_loss(self, name: str, loss: torch.Tensor) -> None:
        self.log(
            name=f'{name}_loss',
            value=loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=1.)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.7,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, targets = batch
        output = self(inputs)

        loss = F.nll_loss(output, targets)
        self._log_loss('training', loss)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, targets = batch
        output = self(inputs)

        loss = F.nll_loss(output, targets)
        self._log_loss('validation', loss)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, targets = batch
        output = self(inputs)

        loss = F.nll_loss(output, targets)
        self._log_loss('test', loss)

        return loss

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


__all__ = [
    'CNN',
]
