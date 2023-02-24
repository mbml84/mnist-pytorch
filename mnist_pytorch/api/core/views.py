from __future__ import annotations

from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


class PredictView(APIView):

    @classmethod
    def post(
            cls,
            request: Request,
    ) -> Response:

        return Response()


__all__ = [
    'PredictView',
]
