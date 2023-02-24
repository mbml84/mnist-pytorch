from __future__ import annotations

import numpy as np
import torch
from api import settings
from PIL import Image

_MODEL = torch.jit.load(settings.WEIGHTS_PATH)
INPUT_SIZE = (28, 28)


def predict(image_stream):
    image = Image.open(image_stream).convert('L')
    image = image.resize(INPUT_SIZE)
    image_array = np.array(image).astype(np.float64) / 255
    tensor_input = torch.tensor(image_array, dtype=torch.float).permute(0, 1).unsqueeze(0).unsqueeze(0)
    prediction = _MODEL(tensor_input)
    return torch.argmax(prediction) + 1


__all__ = [
    'predict',
]
