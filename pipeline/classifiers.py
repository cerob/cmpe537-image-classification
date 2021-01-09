
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import numpy as np

class AdaboostClassifier:

    def __init__(self):
        pass

    def train(self,X, y):
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(X, y)
        clf.score(X, y)
        filename = 'bovw_adaboost.sav'
        print("ADABOOST SAVED")
        pickle.dump(clf, open(filename, 'wb'))
        print(clf.score(X, y))

    def test(self,X, y):
        clf = pickle.load(open('bovw_adaboost.sav', 'rb'))
        # sum=
        # for xt in X:

        # print(np.sum(clf.predict(X) == y) / len(y))
        print(clf.score(X, y))

class SVMClassifier:

    def __init__(self):
        pass

    def train(self,X, y):
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', verbose=True))
        clf.fit(X, y)
        filename = 'bovw_svm.sav'
        print("SVM SAVED")
        print(clf.score(X, y))
        pickle.dump(clf, open(filename, 'wb'))

    def test(self,X, y):
        clf = pickle.load(open('bovw_svm.sav', 'rb'))
        print(clf.score(X, y))