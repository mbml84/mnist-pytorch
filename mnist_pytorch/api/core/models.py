# Create your models here.
from __future__ import annotations

from django.db import models
from django.utils.translation import gettext as _


class Picture(models.Model):

    path = models.FilePathField(verbose_name=_('File path'))
    posted_on = models.DateTimeField(auto_now_add=True, verbose_name=_('Posted on'))
    prediction = models.IntegerField(choices=[(i, i) for i in range(1, 11)], verbose_name=_('Prediction'))
    truth = models.IntegerField(
        choices=[
            (
                i,
                i,
            ) for i in range(
                1,
                11,
            )
        ],
        verbose_name=_('Truth'),
        null=True,
        default=None,
    )

    def __str__(self):
        return f'{self.path} -> {self.prediction}'


__all__ = [
    'Picture',
]
