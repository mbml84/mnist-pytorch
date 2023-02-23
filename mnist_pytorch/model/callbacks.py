from __future__ import annotations

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks import RichProgressBar


def get_default_callbacks() -> list[Callback]:

    return [
        RichModelSummary(),
        RichProgressBar(),
    ]


__all__ = [
    'get_default_callbacks',
]
