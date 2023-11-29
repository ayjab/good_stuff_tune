from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
import optuna
import traceback
import sys

class OptunaTuner:
    @staticmethod
    def tune(trial, model, X_train, y_train, X_valid, y_valid):
        # Your Optuna optimization logic here
        # ...
        return None

class GridSearchTuner:
    @staticmethod
    def tune(model, param_grid, X_train, y_train, X_valid, y_valid):
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

class BayesianTuner:
    @staticmethod
    def tune(model, param_space, X_train, y_train, X_valid, y_valid):
        print("Running Bayesian optimization...", flush=True)
        try:
            bayesian_search = BayesSearchCV(model, param_space, cv=5, verbose=1, n_iter=5, n_jobs=-1, random_state=42)
            bayesian_search.fit(X_train, y_train)
            print("Bayesian optimization done.")
            return bayesian_search
        except Exception as e:
            print(f"Error during Bayesian optimization:")
            traceback.print_exc()
            return None

    @staticmethod
    def get_best_params(tuner):
        return tuner.best_params_

    @staticmethod
    def fit_best_model(tuner, X_train, y_train):
        best_model = tuner.best_estimator_
        best_model.fit(X_train, y_train)
        return best_model

    @staticmethod
    def predict_result(model, X_valid):
        return model.predict(X_valid)