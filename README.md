# Tuning of ML models
This repository is useful for tuning machine learning models for both classification and regression using four methods: Bayesian Optimization, Random Search, Grid Search, and Optuna.

### Usage 
```
git clone https://github.com/ayjab/tune_ml
cd C:\Users\user\good_stuff_tune
python main.py --method --model --problem --data
```
Example:
```
python main.py --method grid_search --model svm --problem classification --data data/iris.csv
```
The data should be uploaded into the data directory, or you can refer to _main.py_ for specific usage.<br>
The configurations for each model and problem type can be found in the directory named cfg. Feel free to modify the hyperparameters and their ranges or to add a specific model in the models directory, wuth respect to its architecture. A snip of a configuration file:

```
model:
  name: SVM  
  problem_type: classification  
  parameters:
    C:
      suggest_type: suggest_float
      min: 1e-5
      max: 10
    kernel:
      suggest_type: suggest_categorical
      choices:
        - "sigmoid"
        - "linear"
        - "poly"
        - "rbf"
    gamma:
      suggest_type: suggest_float
      min: 1e-5
      max: 10
optimization:
  n_trials: 10
  cv: 3
  n_iter: 20
  random_state: 42
  n_jobs: -1
  verbose: 5
  timeout: 3600
  show_progress_bar: True
```
