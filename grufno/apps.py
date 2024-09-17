from django.apps import AppConfig
from grufno.surrogate_model import SurrogateModel

class GrufnoConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'grufno'

    def ready(self):
        model_sat = SurrogateModel()
        model_sat.initialize(model_path="grufno/weights/epoch_49_GRU_780_sat_aug.pth", model_type="sat",
                                input_file="grufno/inference_input_c5.h5", )
        
