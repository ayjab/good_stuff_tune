from .optimization.models import *
#from .regression.models import *
from variables import MODELS #, regression_models

class ModelFactory:
    @staticmethod
    def create_model(problem_type, model_name, **kwargs):
        if problem_type in MODELS.keys() and model_name in MODELS[problem_type].keys():
            model_class = MODELS[problem_type][model_name]
            return model_class(**kwargs)
        else:
            raise ValueError(f"Unknown combination of problem_type and model_name: {problem_type}, {model_name}")
