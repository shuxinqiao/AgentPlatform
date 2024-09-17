from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from grufno.surrogate_model import SurrogateModel
from grufno.db_utils import *
import numpy as np
import json

def gru_inference(request, userID):
    model_sat = SurrogateModel()

    # Extract parameters from request
    percentile = json.loads(request.GET.get('percentile'))
    location = json.loads(request.GET.get('location'))
    rate = json.loads(request.GET.get('rate'))
    bhp = float(json.loads(request.GET.get('bhp')))
    boundary = bool(json.loads(request.GET.get('boundary')))

    prediction = model_sat.inference(percentile=percentile, location=location, rate=rate, bhp=bhp, boundary=boundary)

    description = f"Saturation Prediction by GRU-FNO model with parameters=[percentile:{percentile}, location:{location}, rate:{rate}, bhp:{bhp}, boundary={boundary}. Output shape is {prediction.shape} in numpy array format.]"
    rowid = store_data_sqlite3(filename="test.db", table=userID, data=prediction, type="nparray", description=description)
    
    return JsonResponse({'rowid': rowid})