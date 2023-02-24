from __future__ import annotations

from core import views
from django.urls import path

urlpatterns: list[path] = [
    path('predict/', views.PredictView.as_view(), name='predict'),
]
