import optuna
import yaml
from optimization.optuna_optimization import optimize_random_forest
from models.random_forest_model import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load configuration from config.yml
with open("config/config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

    