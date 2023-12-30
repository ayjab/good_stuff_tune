from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
import optuna
import traceback
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class OptunaTuner:
    @staticmethod
    def tune(model, param_space, X_train, y_train, X_valid, y_valid, dict_params, problem_type):
        OPTUNA_PARAMS = {'n_trials', 'timeout', 'show_progress_bar'}
        relevant_dict_params = {k: v for k, v in dict_params.items() if k in OPTUNA_PARAMS}
        print("Running Optuna optimization...\n", flush=True)
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: OptunaTuner.objective(trial, model, X_train, y_train, X_valid, y_valid, param_space, problem_type), **relevant_dict_params)
            print("\nOptuna optimization done.\n")
            best_trial = study.best_trial
            return best_trial.params
        except Exception as e:
            print(f"Error during Optuna optimization:")
            traceback.print_exc()
            return None
    @staticmethod
    def objective(trial, model, X_train, y_train, X_valid, y_valid, param_space, problem_type):
        params = OptunaTuner.params_trial(trial, param_space)
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = OptunaTuner.predict_result(model, X_valid)
        return mean_absolute_error(y_valid, y_pred)

    @staticmethod
    def params_trial(trial, params_space):
        params = {}        
        for param in params_space.keys():
            suggest_type = params_space[param][-1]
            if suggest_type == "suggest_float":
                params[param] = trial.suggest_float(param, np.float(params_space[param][0]), np.float(params_space[param][1]))
            elif suggest_type == "suggest_int":
                params[param] = trial.suggest_int(param, np.int(params_space[param][0]), np.int(params_space[param][1]))
            elif suggest_type == 'suggest_categorical':
                params[param] = trial.suggest_categorical(param, tuple(params_space[param][0]))
            else:
                raise ValueError(f"Unknown suggest_type: {suggest_type}")
        return params
    
    @staticmethod
    def fit_best_model(model, best_params, X_train, y_train):
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict_result(model, X_valid):
        return model.predict(X_valid)

class GridSearchTuner:
    @staticmethod
    def tune(model, param_space, X_train, y_train, X_valid, y_valid, dict_params, problem_type):
        GRID_PARAMS = {'cv', 'n_jobs', 'verbose', 'n_iter'}
        relevant_dict_params = {k: v for k, v in dict_params.items() if k in GRID_PARAMS}
        print("Running Grid search...\n", flush=True)
        try:
            grid_search = GridSearchCV(model, param_space, **relevant_dict_params)
            grid_search.fit(X_train, y_train)
            print("\n Grid search done.\n")
            return grid_search.best_params_
        except Exception as e:
            print(f"Error during  Grid search:")
            traceback.print_exc()
            return None

    @staticmethod
    def fit_best_model(model, best_params, X_train, y_train):
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict_result(model, X_valid):
        return model.predict(X_valid)

class RandomSearchTuner:
    @staticmethod
    def tune(model, param_space, X_train, y_train, X_valid, y_valid, dict_params, problem_type):
        RANDOM_PARAMS = {'n_iter', 'random_state', 'n_jobs', 'verbose', 'cv'}
        relevant_dict_params = {k: v for k, v in dict_params.items() if k in RANDOM_PARAMS}
        print("Running Random search...\n", flush=True)
        try:
            random_search = RandomizedSearchCV(model, param_space, **relevant_dict_params)
            random_search.fit(X_train, y_train)
            print("\nRandom search done.\n")
            return random_search.best_params_
        except Exception as e:
            print(f"Error during  Random search:")
            traceback.print_exc()
            return None

    @staticmethod
    def fit_best_model(model, best_params, X_train, y_train):
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict_result(model, X_valid):
        return model.predict(X_valid)

class BayesianTuner:
    @staticmethod
    def tune(model, param_space, X_train, y_train, X_valid, y_valid, dict_params, problem_type):
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
    def fit_best_model(model, best_params, X_train, y_train):
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict_result(model, X_valid):
        return model.predict(X_valid)