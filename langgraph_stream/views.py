from django.shortcuts import render

# Create your views here.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from dotenv import load_dotenv

from langgraph_stream.graph import prepare

load_dotenv('.env')

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Bob."),
    ("user", "{input}")
])

llm = ChatOpenAI(model="gpt-4o-mini")


config = {"configurable": {"thread_id": "1"}}

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.userID = self.scope['url_route']['kwargs']['userID']
        self.graph = prepare(llm=llm, userID=self.userID)

        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        print(text_data_json)
        message = text_data_json["message"]

        inputs = {"messages": [message], "userID": self.userID}
        async for event in self.graph.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            tags = event.get("tags", [])
            print(event)
            print("*" * 10)
            if kind == "on_chat_model_stream":
                data = event["data"]["chunk"].content
                id  = event["run_id"]
                
                await self.send(text_data=json.dumps({"event": kind, "data": data, "run_id": id}))

            elif kind == "on_chat_model_start":
                id  = event["run_id"]

                await self.send(text_data=json.dumps({"event": kind, "data": "", "run_id": id}))

            elif kind == "on_chat_model_end":
                id  = event["run_id"]

                await self.send(text_data=json.dumps({"event": kind, "data": "end", "run_id": id}))

        '''
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]

        try:
            # Stream the response
            async for chunk in chain.astream_events({'input': message}, config=config, version="v1", include_names=["Assistant"]):
                #print(chunk)
                if chunk["event"] in ["on_parser_start", "on_parser_stream"]:
                    await self.send(text_data=json.dumps(chunk))

        except Exception as e:
            print(e)
        '''


import sqlite3
import json
from langgraph_stream.db_utils import get_db_json, table_check
from django.http import HttpResponse

def db_view(request, userID):
    if request.method == 'GET':
        table_check(db_name="test.db", table_name=userID)
        return HttpResponse(get_db_json(db_name="test.db", table_name=userID,), content_type="application/json")
    else:
        # send status= 405 which means method not allowed
        return HttpResponse(status=405)
    


from django.conf import settings
import os

def serve_image(request, userID, filename):
    file_path = os.path.join(settings.TEMP_ROOT, userID, filename + '.png')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return HttpResponse(f.read(), content_type="image/png")
    else:
        return HttpResponse("Image not found", status=404)
    


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langgraph_stream.db_utils import *

@csrf_exempt
def file_upload(request, userID):
    if request.method == 'POST' and 'file' in request.FILES:
        file = request.FILES['file']
        temp_dir = os.path.join(settings.BASE_DIR, 'temp', userID)  # Path to /temp/ directory
        # Ensure the directory exists
        os.makedirs(temp_dir, exist_ok=True)

        # Save the file to /temp/ directory
        file_path = os.path.join(temp_dir, file.name)
        
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        description = f"User upload file {file.name} its location has been saved as data in this record."
        store_data_sqlite3(filename="test.db", table=userID, data=f"{file_path}", type="userinput", description=description)

        return JsonResponse({'message': 'File uploaded successfully!', 'file_path': file_path})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)