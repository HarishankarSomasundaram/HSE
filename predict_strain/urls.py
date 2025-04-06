from django.urls import path
from . import views
# from .views import train_model, predict

urlpatterns = [
    path('', views.predict_strain, name='predict_strain'),
    path('train-model/', views.train_model, name='train_model'),
    path('predict-output/', views.predict_output, name='predict_output'),
    path('visualization/', views.visualization, name='visualization'),
]