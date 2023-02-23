from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:

    data_path = Path(__file__).absolute().parents[1] / 'data'
    weights_path = Path(__file__).absolute().parents[1] / 'weights'
    checkpoint_frequency = 20


CONFIG = Config()


__all__ = [
    'CONFIG',
    'Config',
]
