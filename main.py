import optuna
import yaml
from config_reader import ConfigReader
import models.classification.models as clas
import models.regression as regr
from variables import FILE_PATH
from models import ModelFactory
from sklearn.model_selection import train_test_split
from sklearn import datasets
from models import ModelFactory

config_name="config/config_svm.yml"

config_file = ConfigReader(config_name)
optim_infos = config_file.get_optimization_info()
model_info = config_file.get_model_info()
parameters_infos = config_file.get_parameter_ranges()
data_infos = config_file.get_data_paths()

optim_method = optim_infos["method"]
model_name = model_info["name"]
prob_type = model_info["problem_type"]

prob_model_file = FILE_PATH[prob_type]

if __name__ == "__main__":

    model = ModelFactory.create_model(prob_type, model_name)
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    print("_____TRAINING______")
    model.train(X_train, y_train)
    print(model.predict(X_valid))
    print(y_valid)
    print("_____ACCURACY______")
    print(model.accuracy(X_valid, y_valid))   