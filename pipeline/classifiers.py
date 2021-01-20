
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection  import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

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
        clf = pickle.load(open('bovw_svm_grid.sav', 'rb'))
        print(clf.score(X, y))

    def gridSearchTrain(self,X, y):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC())])
        print(pipeline.get_params().keys())

        param_grid = {'svc__C': [0.1, 1, 10, 100],
                      'svc__gamma': [1, 0.1, 0.01, 0.001],
                      'svc__kernel': ['rbf']}
        grid = GridSearchCV(pipeline, param_grid, refit=True, verbose=3, cv=5)

        # fitting the model for grid search
        grid.fit(X, y)

        filename = 'bovw_svm_grid.sav'
        print("SVM SAVED")
        pickle.dump(grid, open(filename, 'wb'))

        # print best parameter after tuning
        print(grid.best_params_)

        # print how our model looks after hyper-parameter tuning
        print(grid.best_estimator_)

    def predict(self,X):
        grid = pickle.load(open('bovw_svm_grid.sav', 'rb'))
        grid_predictions = grid.predict(X)
        return grid_predictions

    def model_scores(self,X,y,le):
        grid = pickle.load(open('bovw_svm_grid.sav', 'rb'))
        grid_predictions = grid.predict(X)
        grid_pred=le.inverse_transform(grid_predictions)
        y_real=le.inverse_transform(y)

        # print classification report
        print(classification_report(y_real, grid_pred))

        # confusion_matrix(y_real, grid_pred)

        titles_options = [("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(grid, X, y,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize,
                                         display_labels=le.classes_,xticks_rotation="vertical")
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)


        plt.show()
        # labels=np.unique(y_real).tolist()
        # cm = confusion_matrix(y_real, grid_pred,labels,normalize="true")
        # print(cm)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(cm)
        # plt.title('Confusion matrix of the classifier')
        # fig.colorbar(cax)
        # ax.set_xticklabels([''] + labels)
        # ax.set_yticklabels([''] + labels)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()

class MLPClassifier:

    def __init__(self):
        pass


    def test(self,X, y):
        clf = pickle.load(open('bovw_mlp_grid.sav', 'rb'))
        print(clf.score(X, y))

    def gridSearchTrain(self,X, y):
        # pipeline = Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("mlp", MLPClassifier())])
        # print(pipeline.get_params().keys())
        mlp=MLPClassifier(random_state=1, max_iter=300)

        param_grid = {
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }
        grid = GridSearchCV(estimator=mlp, param_grid=param_grid, verbose=3, cv=5,n_jobs=-1)


        # fitting the model for grid search
        grid.fit(X, y)

        filename = 'bovw_mlp_grid.sav'
        print("MLP SAVED")
        pickle.dump(grid, open(filename, 'wb'))

        # print best parameter after tuning
        print(grid.best_params_)

        # print how our model looks after hyper-parameter tuning
        print(grid.best_estimator_)

    def model_scores(self,X,y):
        grid = pickle.load(open('bovw_svm_grid.sav', 'rb'))
        grid_predictions = grid.predict(X)

        # print classification report
        print(classification_report(y, grid_predictions))
        print(confusion_matrix())
