from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
    def tune(model, param_space, X_train, y_train, dict_params):
        GRID_PARAMS = {'n_trials', 'cv', 'n_jobs', 'verbose'}
        relevant_dict_params = {k: v for k, v in dict_params.items() if k in GRID_PARAMS}
        print("Running Grid search...\n", flush=True)
        try:
            grid_search = GridSearchCV(model, param_space, **relevant_dict_params)
            grid_search.fit(X_train, y_train)
            print("\n Grid search done.\n")
            return grid_search
        except Exception as e:
            print(f"Error during  Grid search:")
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

class RandomSearchTuner:
    @staticmethod
    def tune(model, param_space, X_train, y_train, dict_params):
        RANDOM_PARAMS = {'n_iter', 'random_state', 'n_jobs', 'verbose', 'cv'}
        relevant_dict_params = {k: v for k, v in dict_params.items() if k in RANDOM_PARAMS}
        print("Running Random search...\n", flush=True)
        try:
            grid_search = RandomizedSearchCV(model, param_space, **relevant_dict_params)
            grid_search.fit(X_train, y_train)
            print("\n Random search done.\n")
            return grid_search
        except Exception as e:
            print(f"Error during  Random search:")
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

class BayesianTuner:
    @staticmethod
    def tune(model, param_space, X_train, y_train, dict_params):
        BAYESIAN_PARAMS = {'n_iter', 'random_state', 'n_jobs', 'verbose', 'cv'}
        relevant_dict_params = {k: v for k, v in dict_params.items() if k in BAYESIAN_PARAMS}
        print("Running Bayesian optimization...\n", flush=True)
        try:
            bayesian_search = BayesSearchCV(model, param_space, **relevant_dict_params)
            bayesian_search.fit(X_train, y_train)
            print("\nBayesian optimization done.\n")
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