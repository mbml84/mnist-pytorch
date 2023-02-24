from __future__ import annotations

import torch
from api import settings

_MODEL = torch.jit.load(settings.WEIGHTS_PATH)


def predict(image_stream):
    ...


__all__ = [
    'predict',
]
