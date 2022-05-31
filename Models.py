import _pickle as pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class Logistic:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        logistic_regression = LogisticRegression(random_state=0)
        logistic_regression.fit(self.X_train, self.y_train)
        filename = 'logistic_model'
        pickle.dump(logistic_regression, open(filename, 'wb'))

class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
        Decision_tree.fit(self.X_train, self.y_train)
        filename = 'dtree_model'
        pickle.dump(Decision_tree, open(filename, 'wb'))

class Svc_kernal:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Svc_kernal = SVC(kernel='linear', random_state=0)
        Svc_kernal.fit(self.X_train,self.y_train)
        filename = 'svc_model'
        pickle.dump(Svc_kernal, open(filename, 'wb'))

class Random_forest:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Random_forest = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0)
        Random_forest.fit(self.X_train,self.y_train)
        filename = 'random_model'
        pickle.dump(Random_forest, open(filename, 'wb'))

class Svc_Polynomial:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        svc_polynomial = SVC(kernel='poly', degree=2)
        svc_polynomial.fit(self.X_train, self.y_train)
        filename = 'polynomial_model'
        pickle.dump(svc_polynomial, open(filename, 'wb'))
