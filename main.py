import argparse
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import datasets
import os
from config_reader import ConfigReader
import yaml
import pandas as pd
from models import ModelFactory
from optimization.init import TuneFactory
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description='Run optimization method.')
parser.add_argument('--method', type=str, choices=['optuna', 'random_search', 'bayesian_optim', 'grid_search'], help='Optimization method to use')
parser.add_argument('--model', type=str, choices=['randomforest', 'svm'], help='Model to use')
parser.add_argument('--problem', type=str, choices=['classification', 'regression'],help='Problem type (classification or regression)')
parser.add_argument('--data', type=str, help='Location of the data file')
args = parser.parse_args()

config_filename = f"cfg/config_{args.problem}_{args.model}.yml"
if not os.path.isfile(config_filename):
    print(f"Error: Configuration file {config_filename} does not exist.")
    exit(1)
config_file = ConfigReader(config_filename)

data_path = args.data 
if not os.path.isfile(data_path):
    print(f"Error: data file {data_path} does not exist.")
    exit(1)
data = pd.read_csv(data_path, delimiter=",")

optim_method = args.method

optim_params = config_file.get_optimization_params()
parameters_infos = config_file.get_parameter_ranges(optim_method)

model_info = config_file.get_model_info()
model_name = model_info["name"]
problem_type = model_info["problem_type"]

X = data.iloc[:, :1]
y = data.iloc[:, -1]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

model = ModelFactory.create_model(problem_type, model_name)
optim_model = TuneFactory.create_model(optim_method)
best_params = optim_model.tune(model=model.model, param_space=parameters_infos, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, dict_params=optim_params, problem_type=problem_type)
print(f"Best Parameters:")
print(best_params)
print("\n")

"""best_model = optim_model.fit_best_model(model.model, best_params, X_train, y_train)
y_pred = optim_model.predict_result(best_model, X_valid)
print("Final accuracy:")
print(accuracy_score(y_valid, y_pred))
print("\n")

model_ = ModelFactory.create_model(problem_type, model_name)
model_.train(X_train, y_train)
print("Non Tuned accuracy:")
print(model_.accuracy(X_valid, y_valid))"""
