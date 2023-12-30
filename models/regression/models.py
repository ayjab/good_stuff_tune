from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge

class SVMRegressor:
    def __init__(self, *args):
        self.model = SVR(*args)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def mse(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return mean_squared_error(y_valid, y_pred)

class RandomForestRegressorWrapper:
    def __init__(self, *args):
        self.model = RandomForestRegressor(*args)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def mse(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return mean_squared_error(y_valid, y_pred)

class XGBoostRegressorWrapper:
    def __init__(self, *args):
        self.model = XGBRegressor(*args)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def mse(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return mean_squared_error(y_valid, y_pred)

class LassoRegressorWrapper:
    def __init__(self, *args):
        self.model = Lasso(*args)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def mse(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return mean_squared_error(y_valid, y_pred)

class RidgeRegressorWrapper:
    def __init__(self, *args):
        self.model = Ridge(*args)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def mse(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return mean_squared_error(y_valid, y_pred)