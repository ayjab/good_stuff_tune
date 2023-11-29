import yaml
import numpy as np
import optuna

class ConfigReader:
    def __init__(self, config_file):
        with open(config_file, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def read(self):
        return self.config

    def get_model_info(self):
        model_info = {
            'name': self.config['model']['name'],
            'problem_type': self.config['model']['problem_type'],
        }
        return model_info

    def get_optimization_info(self):
        optimization_info = {
            'method': self.config['optimization']['method'],
            'n_trials': self.config['optimization']['n_trials']
        }
        return optimization_info

    def get_data_paths(self):
        data_paths = {
            'train_path': self.config['data']['train_path'],
            'test_path': self.config['data']['test_path']
        }
        return data_paths

    def get_parameter_ranges(self):
        parameter_ranges = {}
        for param, param_info in self.config['model']['parameters'].items():
            suggest_type = param_info['suggest_type']
            if suggest_type in ["suggest_float", "suggest_int"]:
                parameter_ranges[(param, suggest_type)] = [np.float(param_info['min']), np.float(param_info['max'])]
            elif suggest_type == 'suggest_categorical':
                parameter_ranges[(param, suggest_type)] = param_info['choices']
            else:
                raise ValueError(f"Unknown suggest_type: {suggest_type}")

        return parameter_ranges

