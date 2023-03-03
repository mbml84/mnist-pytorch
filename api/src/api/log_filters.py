from __future__ import annotations

import logging


class StandardFilter(logging.Filter):
    def filter(self, record):
        return record.levelno <= logging.INFO


class WarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.WARNING


class CriticalFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.ERROR


__all__ = [
    'CriticalFilter',
    'StandardFilter',
    'WarningFilter',
]
