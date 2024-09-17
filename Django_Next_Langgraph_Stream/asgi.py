"""
ASGI config for Django_Next_Langgraph_Stream project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os  
from django.core.asgi import get_asgi_application  
from channels.routing import ProtocolTypeRouter, URLRouter  
from channels.auth import AuthMiddlewareStack  
import langgraph_stream.routing  
  
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Django_Next_Langgraph_Stream.settings')  
  
application = ProtocolTypeRouter({  
  "http": get_asgi_application(),  
  "websocket": AuthMiddlewareStack(  
        URLRouter(  
            langgraph_stream.routing.websocket_urlpatterns  
        )  
    ),  
})