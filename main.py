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
from sklearn.metrics import accuracy_score

# config_name="config/config_svm.yml"
config_name="config/config_rand_for.yml"

config_file = ConfigReader(config_name)

optim_infos = config_file.get_optimization_info()
optim_method = optim_infos["method"]

optim_params = config_file.get_optimization_params()
parameters_infos = config_file.get_parameter_ranges(optim_method)

model_info = config_file.get_model_info()
model_name = model_info["name"]
prob_type = model_info["problem_type"]

data_infos = config_file.get_data_paths()

iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

if __name__ == "__main__":

    print("Model: ", model_name)
    print("Optimization method: ", optim_method)
    print("Class: ", prob_type)
    print("\n")   

    model = ModelFactory.create_model(prob_type, model_name)
    optim_model = TuneFactory.create_model(optim_method)
    tuning = optim_model.tune(model=model.model, param_space=parameters_infos, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, dict_params=optim_params)
    print(f"Best Parameters:")
    print(optim_model.get_best_params(tuning))
    print("\n")

    best_model = optim_model.fit_best_model(tuning, X_train, y_train)
    y_pred = optim_model.predict_result(best_model, X_valid)
    print("Tuned accuracy:")
    print(accuracy_score(y_valid, y_pred))
    print("\n")

    model_ = ModelFactory.create_model(prob_type, model_name)
    model_.train(X_train, y_train)
    print("Non Tuned accuracy:")
    print(model_.accuracy(X_valid, y_valid))
