import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle

import os
import sys
import datetime
import time

from grufno.utils.utils import *
from grufno.utils.lploss import *
from grufno.utils.normalizer import *
from grufno.utils import fno_3d_LKA
from grufno.utils import fno_3d_time


class SurrogateModel:
    # Singleton
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
        return cls._instance
    
    def initialize(self, model_path, model_type, device=None, input_file="inference_input.h5", normalizer_path=None, num_channel=5):
        """
        Surrogate model initialization.
        Define device, model type, dataset, and model itself.
        Args:
            model_path: str, path of weight, model is determined inside class
            model_type: str, which pre-defined model in class to use
            device: torch object, please do not use string, leave None if not sure
            input_file: str, path of inputs contains different permeabilities
        return:
            object: object, initialized model object
        """
        assert model_type in ["sat","pres"], "Model type not support"
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.num_channel = num_channel

        if model_type == "pres":
            self.output_normalizer = self.normalizer(normalizer_path)

        with h5py.File(input_file, 'r') as fin:
            self.input_data = fin["comb_input"][:]

        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()


    def load_model(self, model_path):
        """
        Load model weight from path to selected type
        Args:
            model_path: str, path to weight file .pth
        return:
            model: torch object
        """
        modes = [12,30,30]
        width = 16

        if self.model_type == "sat":
            model = fno_3d_LKA.FNO3d(modes[0], modes[1], modes[2], width, input_channel=self.num_channel, num_cell=4, skip=False)
        elif self.model_type == "pres":
            model = fno_3d_LKA.FNO3d(modes[0], modes[1], modes[2], width, input_channel=self.num_channel, num_cell=4, skip=False)

        model.load_state_dict(torch.load(model_path)['model_state_dict'], strict=True)
        
        return model
    
    def normalizer(self, normalizer_path):
        """
        Normalizer for pressure model.
        Args:
            normalizer_path: str, path of normalizer to use
        return:
            normalizer: object, defined in header normalizers.py
        """
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
            
            if self.device.type == "cuda":
                normalizer.cuda()
            else:
                normalizer.cpu()
        
        return normalizer
    
    def preprocess(self, percentile):
        """
        Take out given permeability input numpy data in shape (B,T,C,Z,X,Y)
        Args:
            percentile: str, in one of p_list
        return:
            data: numpy array
        """
        p_list = ["P20", "P40", "P60", "P80", "P30", "P50", "P25", "P75", "P5", "P10", "P15", "P65", "P95"]
        assert percentile in p_list
        index = p_list.index(percentile)

        current_input = torch.tensor(self.input_data[index:index+1], dtype=torch.float32, device=self.device)

        return current_input
    
    def postprocess(self):
        """
        postprocee, not needed currently
        """
        pass

    def inference(self, percentile=None, location=[(31,28)], rate=[0.5], bhp=60000, boundary=False):
        """
        Main process of inference.
        1. Prepare input data with given permeability, location and rate
        2. model inference
        3. denormalize if using pressure model
        4. return result
        
        Args:
            percentile: str, must in p_list
            location: sets in list, in (x,y) format, from 0 to length-1
            rate: float, 1.0666845 for 0.7MT, use 1e-6 for m3/d
        return:
            current_output: numpy array, output in pressure or saturation
        """
        assert isinstance(location, list), "Location missing, should in [(x1,y1), (x2,y2), ...]"
        for rt in rate:
            assert 0 <= rt <= 2 , "Rate missing or not in range(0,2)"

        
        # Give Permeability and Porosity
        if percentile in ["P20", "P40", "P60", "P80", "P30", "P50", "P25", "P75", "P5", "P10", "P15", "P65", "P95"]:
            current_input = self.preprocess(percentile)
        else:
            current_input = percentile

        
        # Give injection locations and rate
        for i, loc in enumerate(location):
            current_input[0,1:,2,:,loc[0],loc[1]] = rate[i] 

        
        # Give Bhp and Boundary
        current_input[0,:,3,...] = bhp * 1e-5
        current_input[0,:,4,...] = 1 if boundary else -1


        # Inference
        with torch.no_grad():
            current_output = self.model(current_input)

        if self.model_type == "pres":
            current_output = self.output_normalizer.decode(current_output)

        return current_output.cpu().detach().numpy()
