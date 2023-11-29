from models.classification.models import * 
from optimization.models_optim import *
from models.regression.models import *


MODELS = {
    'classification': {
        'SVM': SVMClassifier,
        'RandomForest': RandomForestClassifierWrapper,
        #'LogisticRegression': LogisticRegressionWrapper,
    },
    #'regression': {
    #    'SVM': SVMRegressor,
    #    'RandomForest': RandomForestRegressorWrapper,
    #    'LinearRegression': LinearRegressionWrapper,
    #},
}

OPTIM_MODELS = {"optuna": OptunaTuner,
        "grid_search": GridSearchTuner,
        "bayesian_optim": BayesianTuner
}

FILE_PATH = {
    "classification": "models/classification", 
    "regression": "models/regression"
    }