from django.urls import path, re_path
from . import views

urlpatterns = [
    #path('grufno/sat/<int:userID>/', views.gru_inference),
    re_path(r'grufno/direct/sat/(?P<userID>\w+)', views.gru_inference_direct),
    re_path(r'grufno/sat/(?P<userID>\w+)', views.gru_inference),
]