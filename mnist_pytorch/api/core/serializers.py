from __future__ import annotations

from rest_framework import serializers

from mnist_pytorch.api.core import models


class PictureSerializer(serializers.ModelSerializer):

    class Meta:
        model = models.Picture
        read_only_fields = ['posted_on']
        fields = '__all__'


__all__ = [
    'PictureSerializer',
]
