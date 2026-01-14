from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('get_response/', views.get_response, name='get_response'),
    path('scrape_api/', views.scrape_api, name='scrape_api'),
    path('train_api/', views.train_api, name='train_api'),
    path('sentiment_api/', views.sentiment_api, name='sentiment_api'),  # NEW
]