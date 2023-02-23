from __future__ import annotations

import fire
from model.training import train


class Runner:

    @classmethod
    def train(
        cls,
        accelerator: str = 'cpu',
        devices: int = 1,
        max_epochs: int = 200,
        log_every_n_steps: int = 50,
        batch_size: int = 32,
        checkpoint_path: str | None = None,
    ):

        train(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
        )


if __name__ == '__main__':
    fire.Fire(Runner)


__all__ = [
    'Runner',
]
