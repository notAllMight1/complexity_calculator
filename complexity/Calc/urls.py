from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Main page view
    path('predict/', views.predict, name='predict'),  # API view for predictions
]
