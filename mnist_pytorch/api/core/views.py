from __future__ import annotations

from core.prediction import predict
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


class PredictView(APIView):

    @classmethod
    def post(
            cls,
            request: Request,
    ) -> Response:
        image_stream = request.FILES['image']
        prediction = predict(image_stream)
        return Response(
            {
                'prediction': prediction,
            },
        )


__all__ = [
    'PredictView',
]
