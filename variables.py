from models.classification.models import * #, RandomForestClassifierWrapper, LogisticRegressionWrapper
#from models.regression.models import SVMRegressor, RandomForestRegressorWrapper, LinearRegressionWrapper


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

FILE_PATH = {
    "classification": "models/classification", 
    "regression": "models/regression"
    }