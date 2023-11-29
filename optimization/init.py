from .models_optim import *
from variables import OPTIM_MODELS 

class TuneFactory:
    @staticmethod
    def create_model(optim_method, *args, **kwargs):
        if optim_method in OPTIM_MODELS.keys():
            optim_tuner = OPTIM_MODELS[optim_method]
            return optim_tuner(*args, **kwargs)
        else:
            raise ValueError(f"Unknown optim_method: {optim_method}")
