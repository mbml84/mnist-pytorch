# Create your models here.
from __future__ import annotations

from django.db import models
from django.utils.translation import gettext as _


def upload_to(instance, filename):
    return f'images/{filename}'


class Picture(models.Model):

    CHOICES = [(i, i) for i in range(1, 11)]

    image = models.ImageField(upload_to=upload_to, blank=True, null=True, verbose_name=_('Image'))
    posted_on = models.DateTimeField(auto_now_add=True, verbose_name=_('Posted on'))
    prediction = models.IntegerField(choices=CHOICES, verbose_name=_('Prediction'), null=True)
    truth = models.IntegerField(
        choices=CHOICES,
        verbose_name=_('Truth'),
        null=True,
        default=None,
    )

    def __str__(self):
        return f'{self.image.name} -> {self.prediction}'


__all__ = [
    'Picture',
]
