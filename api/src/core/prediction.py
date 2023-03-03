from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from api import settings

__LOGGER__ = logging.getLogger(__name__)


def _get_model():
    try:
        return torch.jit.load(settings.WEIGHTS_PATH)
    except Exception:
        __LOGGER__.exception(
            'No model has been loaded. Prediction will always be -1',
        )
        return None


_MODEL = _get_model()
INPUT_SIZE = (28, 28)


def predict(image_stream) -> int:
    if _MODEL:
        image = Image.open(image_stream).convert('L')
        image = image.resize(INPUT_SIZE)
        image_array = np.array(image).astype(np.float64) / 255
        tensor_input = torch.tensor(image_array, dtype=torch.float).permute(
            0, 1,
        ).unsqueeze(0).unsqueeze(0)
        prediction = _MODEL(tensor_input)
        prediction = torch.argmax(prediction).item() + 1
    else:
        __LOGGER__.warning(
            'No model has been loaded. Prediction will always be -1',
        )
        prediction = -1
    return prediction


__all__ = [
    'predict',
]
