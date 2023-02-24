from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.utilities.types import STEP_OUTPUT

from mnist_pytorch.model import utils


class CNN(torch.nn.Module):

    MODEL_NAME = 'mnist-cnn'

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

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


class DistributedCNN(LightningModule):

    MODEL_NAME = 'mnist-cnn'

    def __init__(self, model: CNN):
        super().__init__()
        self.model = model
        self._metrics = {
            stage: utils.ClassificationMetrics(parent=self, stage=stage) for stage in list(utils.Stage)
        }
        self.example_input_array = torch.Tensor(32, 1, 28, 28)

    def _log_loss(self, name: str, loss: torch.Tensor) -> None:
        self.log(
            name=f'{name}_loss',
            value=loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

    def _execute_step(self, stage: utils.Stage, batch, batch_idx: int) -> dict[str, torch.Tensor]:
        inputs, targets = batch
        output = self(inputs)

        loss = F.nll_loss(output, targets)
        self._log_loss(stage.value, loss)
        self._metrics[stage].update(output, targets)
        return {'loss': loss, 'preds': output, 'target': targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=1.)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.7,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        return self._execute_step(utils.Stage.TRAIN, batch, batch_idx)

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        self._metrics[utils.Stage.TRAIN].log_step_metrics()
        return super().training_step_end(step_output)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._metrics[utils.Stage.TRAIN].log_epoch_metrics()

    def validation_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        return self._execute_step(utils.Stage.VALIDATION, batch, batch_idx)

    def validation_step_end(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        self._metrics[utils.Stage.VALIDATION].log_step_metrics()
        return super().validation_step_end(*args, **kwargs)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]) -> None:
        self._metrics[utils.Stage.VALIDATION].log_epoch_metrics()

    def test_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        return self._execute_step(utils.Stage.TEST, batch, batch_idx)

    def test_step_end(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        self._metrics[utils.Stage.TEST].log_step_metrics()
        return super().test_step_end(*args, **kwargs)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]) -> None:
        self._metrics[utils.Stage.TEST].log_epoch_metrics()

    def forward(self, x):
        return self.model(x)


__all__ = [
    'CNN',
    'DistributedCNN',
]
