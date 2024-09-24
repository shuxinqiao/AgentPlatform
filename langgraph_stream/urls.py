from django.urls import path, re_path
from . import views  


urlpatterns = [  
    #re_path(r'ws/chat/(?P<userID>\w+)', views.ChatConsumer.as_asgi()),
    re_path(r'db/(?P<userID>\w+)', views.db_view),
    re_path(r'temp/(?P<userID>\w+)/(?P<filename>\w+)', views.serve_image),
    re_path(r'upload/(?P<userID>\w+)', views.file_upload),
]