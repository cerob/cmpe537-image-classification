import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import walk
import os
import random
import pdb
import pickle

from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection  import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

#because of problems in vlfeat import I used the fisher vector code from
#this pipeline is
#https://gist.github.com/danoneata/9927923
def fisher_vector(xx, gmm):

    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

def load_folder(path):
    all_files = []
    classes=[]
    docs=[]
    for (dirpath, dirnames, filenames) in walk(path):
        for file in filenames:
            dirname=dirpath.replace(path+"\\","")
            all_files.append(os.path.join(dirpath,file))
            classes.append(dirname)

    return all_files,classes

def orb_descriptor(file,nfeatures=500):
    img = cv2.imread(file)

    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    if des is None:
        return []
    return des

def cluster_gmm(features,filename,K):

    gmm = GMM(n_components=K, covariance_type='diag',verbose=2,max_iter=1000,verbose_interval=1)
    gmm.fit(features)

    print("GMM FIT")
    # fv = fisher_vector(features[-1], gmm)
    # gmm.predict_proba()
    print("GMM SAVED")
    pickle.dump(gmm, open(filename, 'wb'))

def prepare_set(images, classes,le):
    gmm = pickle.load(open('gmm_500_32.sav', 'rb'))
    x=[]
    y=[]
    problematic_indices=[]
    for i,image in enumerate(images):
        points=orb_descriptor(image,500)
        if len(points)==0:
            problematic_indices.append(i)
        else:
            fv=fisher_vector(points,gmm)
            x.append(fv)


    y = le.transform(classes)
    y=np.delete(y,problematic_indices)
    X=np.stack(x,axis=0)
    return X,y

def train(X,y):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)
    clf.score(X, y)
    filename = 'adaboost_classifier.sav'
    print("ADABOOST SAVED")
    pickle.dump(clf, open(filename, 'wb'))

def test(X,y):
    clf = pickle.load(open('adaboost_classifier.sav', 'rb'))
    # sum=
    # for xt in X:

    print(np.sum(clf.predict(X) == y) / len(y))
    print(clf.score(X, y))

def train_svm(X,y):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',verbose=True))
    clf.fit(X, y)
    clf.score(X, y)
    filename = 'svm_classifier.sav'
    print("SVM SAVED")
    pickle.dump(clf, open(filename, 'wb'))

def test_svm(X,y):
    clf = pickle.load(open('fisher_svm_grid.sav', 'rb'))
    # sum=
    # for xt in X:

    # print(np.sum(clf.predict(X) == y) / len(y))
    print(clf.score(X, y))

def gridSearchTrain(X, y):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC())])
    print(pipeline.get_params().keys())

    param_grid = {'svc__C': [10, 100, 1000],
                  'svc__gamma': [0.001, 0.0001],
                  'svc__kernel': ['rbf']}
    grid = GridSearchCV(pipeline, param_grid, refit=True, verbose=3, cv=5,n_jobs=4)

    # fitting the model for grid search
    grid.fit(X, y)

    filename = 'fisher_svm_grid.sav'
    print("SVM SAVED")
    pickle.dump(grid, open(filename, 'wb'))

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

def model_scores(X,y,le):
    grid = pickle.load(open('fisher_svm_grid.sav', 'rb'))
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



if __name__ == '__main__':
    train_images, train_classes=load_folder("C:/Users/yusuf/Desktop/MS/CMPE 537/Caltech20/training")
    test_images, test_classes=load_folder("C:/Users/yusuf/Desktop/MS/CMPE 537/Caltech20/testing")
    descriptions=[]
    le=pickle.load(open('label_encoder.sav', 'rb'))
    # print(len(train_images))
    # for image in train_images:
    #     descriptions.extend(orb_descriptor(image,500))
    #
    # print("DESCRIPTORS' SIZE: " +str(len(descriptions)))
    # cluster_gmm(features=descriptions,filename = 'gmm_500_32.sav',K=32)


    # loaded_model = pickle.load(open('gmm_all_128.sav', 'rb'))
    # prob=loaded_model.predict_proba(descriptions[:1])
    # print(prob)
    x_train,y_train=prepare_set(train_images, train_classes,le)
    x_test,y_test=prepare_set(test_images, test_classes,le)
    # train(x_train,y_train)
    # test(x_train,y_train)
    # test(x_test,y_test)

    gridSearchTrain(x_train,y_train)
    test_svm(x_train,y_train)
    test_svm(x_test,y_test)
    model_scores(x_test,y_test,le)
