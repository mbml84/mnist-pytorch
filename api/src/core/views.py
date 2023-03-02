from __future__ import annotations

from copy import deepcopy

from core import models
from core import serializers
from core.prediction import predict
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


class PredictView(APIView):

    @classmethod
    def get(
            cls,
            request: Request,
    ) -> Response:
        query_params = request.query_params
        pictures = models.Picture.objects.filter(**query_params)
        return Response(
            serializers.PictureSerializer(pictures, many=True).data,
            status=status.HTTP_200_OK,
        )

    @classmethod
    def post(
            cls,
            request: Request,
    ) -> Response:
        image_stream = request.FILES['image']
        prediction = predict(deepcopy(image_stream))
        data = {
            'image': image_stream,
            'prediction': prediction,
        }
        serializer = serializers.PictureSerializer(data=data)
        if serializer.is_valid(raise_exception=True):
            instance = serializer.create(data)
            instance.save()
            response = Response(
                {'prediction': prediction},
                status=status.HTTP_200_OK,
            )
        else:
            response = Response(status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        return response


__all__ = [
    'PredictView',
]
