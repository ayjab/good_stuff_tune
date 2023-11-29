from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def accuracy(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return accuracy_score(y_valid, y_pred)

class RandomForestClassifierWrapper:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def accuracy(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        return accuracy_score(y_valid, y_pred)


