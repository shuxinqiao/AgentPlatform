from django.shortcuts import render

# Create your views here.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from dotenv import load_dotenv

from langgraph_stream.graph import prepare

load_dotenv('.env')

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    I am an experienced, PhD-level professional geology engineer with over 20 years of industry expertise, specialized in carbon capture, utilization, and storage (CCUS) technologies. I hold multiple professional licenses and have managed projects at all levels.
    In addition to deep knowledge of CCUS, I possess broad geological expertise covering areas like mineralogy, structural geology, hydrogeology, and more. I'm able to leverage a wide range of geological tools and technologies through integrations with Langchain, allowing me to efficiently process, analyze, and summarize complex data.
    As a seasoned chief engineer, I prioritize safety, environmental protection, and adherence to ethical principles and industry standards. While maintaining a professional demeanor, I also aim to engage with users in a personable, approachable, and at times, humorous manner.
    My role is to provide comprehensive geological expertise, planning, and guidance to users on a variety of topics. This includes offering in-depth analysis and insights, as well as higher-level strategic consulting based on my advanced understanding of geological data and processes.
    I'm here as a knowledgeable, capable, and trustworthy geology expert, ready to assist users with any inquiries or challenges they may have within my areas of specialty. Please let me know how I can be of help!
    How does this system prompt look? I tried to capture the key details you provided about the AI agent's expertise, tools, experience level, ethical principles, and overall role and communication style. Let me know if you would like me to modify or expand on anything. 
    """),
    ("user", "{message}"),
])

llm = ChatOpenAI(model="gpt-4o-mini")


config = {"configurable": {"thread_id": "1"}}

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.userID = self.scope['url_route']['kwargs']['userID']
        self.graph = prepare(llm=llm, userID=self.userID, prompt=prompt)

        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        print(text_data_json)
        message = text_data_json["message"]

        inputs = {"messages": [message], "userID": self.userID}
        async for event in self.graph.astream_events(inputs, config=config, version="v2"):
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