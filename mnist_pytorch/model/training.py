from __future__ import annotations

from datetime import datetime

import fire
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from mnist_pytorch.config import CONFIG
from mnist_pytorch.model import callbacks
from mnist_pytorch.model.cnn import CNN
from mnist_pytorch.model.datamodule import MNISTDataModule


def _setup(
        accelerator: str,
        devices: int,
        batch_size: int,
        model: CNN,
        max_epochs: int,
        log_every_n_steps: int,
        checkpoint_path: str | None,
):
    if checkpoint_path is not None:
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path, model=model,
        )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=TensorBoardLogger(save_dir='logs'),
        log_every_n_steps=log_every_n_steps,
        max_epochs=max_epochs,
        gradient_clip_algorithm='norm',
        callbacks=callbacks.get_default_callbacks(),
    )

    return trainer, model, MNISTDataModule(batch_size=batch_size)


class Runner:

    @classmethod
    def train(
        cls,
        accelerator: str = 'cpu',
        devices: int = 1,
        max_epochs: int = 1,
        log_every_n_steps: int = 50,
        batch_size: int = 32,
        checkpoint_path: str | None = None,
    ):
        trainer, model, datamodule = _setup(
            model=CNN(),
            batch_size=batch_size,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            checkpoint_path=checkpoint_path,
        )

        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

        trainer.test(
            model=model,
            datamodule=datamodule,
        )

        trainer.validate(
            model=model,
            datamodule=datamodule,
        )

        filename = f'{CNN.MODEL_NAME}-{datetime.utcnow()}-{max_epochs=}-{batch_size=}.pt'
        model_jit = torch.jit.script(model)
        torch.jit.save(model_jit, CONFIG.weights_path / filename)


if __name__ == '__main__':
    fire.Fire(Runner)


__all__ = [
    'Runner',
]
