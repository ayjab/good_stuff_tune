import optuna
from models.classification.models import SVMClassifier, RandomForestClassifierWrapper, LogisticRegressionWrapper

def objective(trial):
    model_name = "SVM"  # Change this based on the model you want to optimize
    if model_name == "SVM":
        model = SVMClassifier(
            C=trial.suggest_loguniform("C", 1e-5, 1e5),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
            gamma=trial.suggest_loguniform("gamma", 1e-5, 1e5),
        )
    elif model_name == "RandomForest":
        model = RandomForestClassifierWrapper(
            n_estimators=trial.suggest_int("n_estimators", 10, 1000),
            max_depth=trial.suggest_int("max_depth", 1, 32),
        )
    elif model_name == "LogisticRegression":
        model = LogisticRegressionWrapper(
            C=trial.suggest_loguniform("C", 1e-5, 1e5),
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Train the model
    model.train(X_train, y_train)

    # Evaluate the model using accuracy
    accuracy = model.accuracy(X_valid, y_valid)

    return accuracy

# Set up Optuna study and optimize the objective function
# ...
