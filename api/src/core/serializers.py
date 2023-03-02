from __future__ import annotations

from core import models
from rest_framework import serializers


class PictureSerializer(serializers.ModelSerializer):
    image_url = serializers.ImageField(required=False)

    class Meta:
        model = models.Picture
        read_only_fields = ['posted_on']
        fields = '__all__'


__all__ = [
    'PictureSerializer',
]
