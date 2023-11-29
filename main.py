import optuna
import yaml
from config_reader import ConfigReader
import models.classification.models as clas
import models.regression as regr
from models import ModelFactory
from sklearn.model_selection import train_test_split
from sklearn import datasets
from models import ModelFactory
from optimization.init import TuneFactory

config_name="config/config_svm.yml"

config_file = ConfigReader(config_name)
optim_infos = config_file.get_optimization_info()
model_info = config_file.get_model_info()
parameters_infos = config_file.get_parameter_ranges()
data_infos = config_file.get_data_paths()

optim_method = optim_infos["method"]
model_name = model_info["name"]
prob_type = model_info["problem_type"]

param_grid_svm = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

if __name__ == "__main__":

    model = ModelFactory.create_model(prob_type, model_name)
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    optim_model = TuneFactory.create_model(optim_method)
    bayesian_optimization = optim_model.tune(model.model, param_grid_svm, X_train, y_train, X_valid, y_valid)

    print(optim_model.get_best_params(bayesian_optimization))
    print("DONE")
"""    best_model = optim_model.fit_best_model(bayesian_optimization, X_train, y_train)
    y_pred = BayesianTuner.predict(best_model, X_valid)

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy with Best Parameters: {accuracy}")"""