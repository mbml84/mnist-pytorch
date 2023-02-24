from __future__ import annotations

from django.urls import path

from mnist_pytorch.api.core import views

urlpatterns: list[path] = [
    path('predict/', views.PredictView.as_view(), name='predict'),
]
