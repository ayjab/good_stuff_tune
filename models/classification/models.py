from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class SVMClassifier:
    def __init__(self, *args):
        self.model = SVC(*args)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def accuracy(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return accuracy_score(y_valid, y_pred)

class RandomForestClassifierWrapper:
    def __init__(self, *args):
        self.model = RandomForestClassifier(*args)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def accuracy(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return accuracy_score(y_valid, y_pred)


