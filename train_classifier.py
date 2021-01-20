from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
import os


def train_svm(x, y, param_list):
    param_list_str = [str(i) for i in param_list.values()]
    model_name = 'svm' + '_'.join(param_list_str) + '.pkl'
    if os.path.exists('./' + model_name):
        with open('./' + model_name, 'rb') as f:
            svm = pickle.load(f)
    else:
        svm = SGDClassifier(max_iter=param_list['max_iter'], learning_rate=param_list['learning_rate'], eta0=param_list['eta0'])
        svm.fit(x, y.squeeze())
        with open('./' + model_name, 'wb') as f:
            pickle.dump(svm, f)
    return svm


def cross_validation(x, y, param_list):
    model_name = 'svm_5fold_validation.pkl'
    if os.path.exists('./' + model_name):
        with open('./' + model_name, 'rb') as f:
            classifier = pickle.load(f)
    else:
        svm = SGDClassifier()
        classifier = GridSearchCV(svm, param_list)
        classifier.fit(x, y)
        with open('./' + model_name, 'wb') as f:
            pickle.dump(svm, f)
    return classifier