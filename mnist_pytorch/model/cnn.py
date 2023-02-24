from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.utilities.types import STEP_OUTPUT


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
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.train_precision = torchmetrics.Precision(task='multiclass', num_classes=10)
        self.valid_precision = torchmetrics.Precision(task='multiclass', num_classes=10)
        self.test_precision = torchmetrics.Precision(task='multiclass', num_classes=10)
        self.train_recall = torchmetrics.Recall(task='multiclass', num_classes=10)
        self.valid_recall = torchmetrics.Recall(task='multiclass', num_classes=10)
        self.test_recall = torchmetrics.Recall(task='multiclass', num_classes=10)
        self.train_f1_score = torchmetrics.F1Score(task='multiclass', num_classes=10)
        self.valid_f1_score = torchmetrics.F1Score(task='multiclass', num_classes=10)
        self.test_f1_score = torchmetrics.F1Score(task='multiclass', num_classes=10)

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

    def training_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        inputs, targets = batch
        output = self(inputs)

        loss = F.nll_loss(output, targets)
        self._log_loss('training', loss)
        self.train_acc.update(output, targets)
        self.train_precision.update(output, targets)
        self.train_recall.update(output, targets)
        self.train_f1_score.update(output, targets)
        return {'loss': loss, 'preds': output, 'target': targets}

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        self.log('train_accuracy', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=False)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=False)
        self.log('train_f1_score', self.train_f1_score, on_step=True, on_epoch=False)
        return super().training_step_end(step_output)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log('train_accuracy', self.train_acc, on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True)
        self.log('train_f1_score', self.train_f1_score, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        inputs, targets = batch
        output = self(inputs)

        loss = F.nll_loss(output, targets)
        self._log_loss('validation', loss)
        self.valid_acc.update(output, targets)
        self.valid_precision.update(output, targets)
        self.valid_recall.update(output, targets)
        self.valid_f1_score.update(output, targets)
        return {'loss': loss, 'preds': output, 'target': targets}

    def validation_step_end(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        self.log('valid_accuracy', self.valid_acc, on_step=True, on_epoch=False)
        self.log('valid_precision', self.valid_precision, on_step=True, on_epoch=False)
        self.log('valid_recall', self.valid_recall, on_step=True, on_epoch=False)
        self.log('valid_f1_score', self.valid_f1_score, on_step=True, on_epoch=False)
        return super().validation_step_end(*args, **kwargs)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]) -> None:
        self.log('valid_accuracy', self.valid_acc, on_step=False, on_epoch=True)
        self.log('valid_precision', self.valid_precision, on_step=False, on_epoch=True)
        self.log('valid_recall', self.valid_recall, on_step=False, on_epoch=True)
        self.log('valid_f1_score', self.valid_f1_score, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        inputs, targets = batch
        output = self(inputs)

        loss = F.nll_loss(output, targets)
        self._log_loss('test', loss)
        self.test_acc.update(output, targets)
        self.test_precision.update(output, targets)
        self.test_recall.update(output, targets)
        self.test_f1_score.update(output, targets)

        return {'loss': loss, 'preds': output, 'target': targets}

    def test_step_end(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        self.log('test_accuracy', self.test_acc, on_step=True, on_epoch=False)
        self.log('test_precision', self.test_precision, on_step=True, on_epoch=False)
        self.log('test_recall', self.test_recall, on_step=True, on_epoch=False)
        self.log('test_f1_score', self.test_f1_score, on_step=True, on_epoch=False)
        return super().test_step_end(*args, **kwargs)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]) -> None:
        self.log('test_accuracy', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test_f1_score', self.test_f1_score, on_step=False, on_epoch=True)

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
