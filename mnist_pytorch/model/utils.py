from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall


class Stage(Enum):

    TRAIN = 'train'
    VALIDATION = 'valid'
    TEST = 'test'


@dataclass(frozen=True)
class ClassificationMetrics:
    parent: LightningModule
    stage: Stage

    def __post_init__(self):
        # Needed to work with LightningModule
        self.parent.__setattr__(f'{self.stage.value}_accuracy', MulticlassAccuracy(num_classes=10))
        self.parent.__setattr__(f'{self.stage.value}_precision', MulticlassPrecision(num_classes=10))
        self.parent.__setattr__(f'{self.stage.value}_recall', MulticlassRecall(num_classes=10))
        self.parent.__setattr__(f'{self.stage.value}_f1_score', MulticlassF1Score(num_classes=10))

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.parent.__getattr__(f'{self.stage.value}_accuracy').update(predictions, targets)
        self.parent.__getattr__(f'{self.stage.value}_precision').update(predictions, targets)
        self.parent.__getattr__(f'{self.stage.value}_recall').update(predictions, targets)
        self.parent.__getattr__(f'{self.stage.value}_f1_score').update(predictions, targets)

    def log_step_metrics(self) -> None:
        kwargs = {'on_step': True, 'on_epoch': False}
        self.parent.log(
            f'{self.stage.value}_accuracy',
            self.parent.__getattr__(f'{self.stage.value}_accuracy'),
            **kwargs,
        )
        self.parent.log(
            f'{self.stage.value}_precision',
            self.parent.__getattr__(f'{self.stage.value}_precision'),
            **kwargs,
        )
        self.parent.log(f'{self.stage.value}_recall', self.parent.__getattr__(f'{self.stage.value}_recall'), **kwargs)
        self.parent.log(
            f'{self.stage.value}_f1_score',
            self.parent.__getattr__(f'{self.stage.value}_f1_score'),
            **kwargs,
        )

    def log_epoch_metrics(self) -> None:
        kwargs = {'on_step': False, 'on_epoch': True}
        self.parent.log(
            f'{self.stage.value}_accuracy',
            self.parent.__getattr__(f'{self.stage.value}_accuracy'),
            **kwargs,
        )
        self.parent.log(
            f'{self.stage.value}_precision',
            self.parent.__getattr__(f'{self.stage.value}_precision'),
            **kwargs,
        )
        self.parent.log(f'{self.stage.value}_recall', self.parent.__getattr__(f'{self.stage.value}_recall'), **kwargs)
        self.parent.log(
            f'{self.stage.value}_f1_score',
            self.parent.getattr(f'{self.stage.value}_f1_score'),
            **kwargs,
        )


__all__ = [
    'ClassificationMetrics',
    'Stage',
]
