import numpy as np
from spectral_clustering import spectral_clustering, k_means, assign2cluster
from bag_of_words import bag_of_words
from train_classifier import train_svm
from image_descriptor import feature_and_label_extraction_for_nonflat, feature_and_label_extraction_for_flat
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def hog_spectral_bow_svm(cluster_size, svm_param_list, dir='../Caltech20/'):
    dataset = 'training'
    features, labels = feature_and_label_extraction_for_nonflat(dir, dataset, cell_size=32) # cell size has to be 32 because of spectral clustering
    no_images = features.shape[0]
    cluster_model, centroids = spectral_clustering(features.reshape((-1, 36)), cluster_size)
    cluster_labels = cluster_model.labels_.reshape((no_images, -1))
    hists = bag_of_words(cluster_labels, cluster_size)
    classifier = train_svm(hists, labels, svm_param_list)
    print('Training Accuracy: ' + str(classifier.score(hists, labels)))
    return classifier, centroids

def hog_kmeans_bow_svm(cluster_size, svm_param_list, cell_size, dir='../Caltech20/'):
    dataset = 'training'
    features, labels = feature_and_label_extraction_for_nonflat(dir, dataset, cell_size=cell_size)
    no_images = features.shape[0]
    cluster_model = k_means(features.reshape((-1, 36)), cell_size, cluster_size=cluster_size)
    cluster_labels = cluster_model.labels_.reshape((no_images, -1))
    hists = bag_of_words(cluster_labels, cluster_size)
    classifier = train_svm(hists, labels, svm_param_list)
    print('Training Accuracy: ' + str(classifier.score(hists, labels)))
    return classifier, cluster_model


def hog_svm(svm_param_list, cell_size, dir='../Caltech20/'):
    dataset = 'training'
    features, labels = feature_and_label_extraction_for_flat(dir, dataset, cell_size)
    classifier = train_svm(features, labels, svm_param_list)
    print('Training Accuracy: ' + str(classifier.score(features, labels)))
    return classifier


def evaluation(param_list, model, flat, cluster_type, cluster_model, cluster_no):
    dataset = 'testing'
    if flat:
        x_test, y_test = feature_and_label_extraction_for_flat('../Caltech20/', dataset, param_list['cell'])

    else:
        x_test, y_test = feature_and_label_extraction_for_nonflat('../Caltech20/', dataset, param_list['cell'])
        no_images = x_test.shape[0]
        x_test = x_test.reshape((-1, 36))
        if cluster_type == 'spectral':
            cluster_labels = assign2cluster(cluster_model, x_test)
        else:
            cluster_labels = cluster_model.predict(x_test)

        cluster_labels = cluster_labels.reshape((no_images, -1))
        x_test = bag_of_words(cluster_labels, cluster_no)

    y_predict = model.predict(x_test)
    report = classification_report(y_test.squeeze(), y_predict, zero_division=1)
    confusion = confusion_matrix(y_true=y_test, y_pred=y_predict)
    return report, confusion




if __name__ == '__main__':
    cluster_size = 64
    param_list1 = {'max_iter': 100,
                  'learning_rate': 'optimal',
                  'eta0': 0,
                  'cell': 32,
                   'name': 'spectral'}
    model1, centroids = hog_spectral_bow_svm(cluster_size, param_list1)

    param_list2 = {'max_iter': 800,
                  'learning_rate': 'optimal',
                  'eta0': 0,
                  'cell': 8,
                   'name': 'kmeans'}
    model2, cluster_model = hog_kmeans_bow_svm(cluster_size, param_list2, cell_size=8)

    param_list3 = {'max_iter': 300,
                  'learning_rate': 'optimal',
                  'eta0': 0,
                  'cell': 8,
                  'name': 'direct'}
    model3 = hog_svm(param_list3, cell_size=8)

    #evaluate
    report1, confusion1 = evaluation(param_list1, model1, False, 'spectral', centroids, cluster_size)
    report2, confusion2 = evaluation(param_list2, model2, False, 'kmeans', cluster_model, cluster_size)
    report3, confusion3 = evaluation(param_list3, model3, True, None, None, None)
    print('Spectral')
    print(report1)
    print('Kmeans')
    print(report2)
    print('Direct HOG')
    print(report3)

    plt.figure(1)
    plt.imshow(confusion1)
    plt.title('Confusion Matrix for HOG-Spectral-BOW-SVM')
    plt.savefig('spectral_confusion.png')

    plt.figure(2)
    plt.imshow(confusion2)
    plt.title('Confusion Matrix for HOG-Kmeans-BOW-SVM')
    plt.savefig('kmeans_confusion.png')

    plt.figure(3)
    plt.imshow(confusion3)
    plt.title('Confusion Matrix for HOG-SVM')
    plt.savefig('direct_hog_confusion.png')

    plt.show()