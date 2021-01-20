from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import numpy as np
import pickle
import os


def spectral_clustering(features, cluster_size=64, affinity='rbf', assign_labels='kmeans'):
    model_name = 'spectral_' + affinity + '_' + assign_labels + '_' + str(cluster_size) + '.pkl'
    if os.path.exists('./' + model_name):
        with open('./' + model_name, 'rb') as f:
            cluster_model = pickle.load(f)
    else:
        cluster_model = SpectralClustering(n_clusters=cluster_size, affinity=affinity, assign_labels=assign_labels)
        cluster_model.fit(features)

        with open('./' + model_name, 'wb') as f:
            pickle.dump(cluster_model, f)
    data_labels = cluster_model.labels_
    centroids = np.zeros((cluster_size, features.shape[1]))
    for i in range(cluster_size):
        index = np.where(data_labels==i)
        centroids[i] = np.mean(features[index], axis=0)
    return cluster_model, centroids


def k_means(features, cell , cluster_size=64, max_iter=10):
    model_name = 'kmeans_' + str(cell) + '_' + str(max_iter) + '_' + str(cluster_size) + '.pkl'
    if os.path.exists('./' + model_name):
        with open('./' + model_name, 'rb') as f:
            cluster_model = pickle.load(f)
    else:
        cluster_model = KMeans(n_clusters=cluster_size, max_iter=max_iter)
        cluster_model.fit(features)

        with open('./' + model_name, 'wb') as f:
            pickle.dump(cluster_model, f)
    return cluster_model

def assign2cluster(centroids, data):
    cluster_labels = np.zeros((data.shape[0], 1))
    for i in range(data.shape[0]):
        dist = np.sqrt(np.sum((centroids - np.tile(data[i,:], (centroids.shape[0],1)))**2, axis=1))
        index = np.argmin(dist).astype(dtype=int)
        cluster_labels[i] = index
    return cluster_labels

if __name__ == '__main__':

    features = np.load('hog_training_features.npy').flatten()
    cluster_model = spectral_clustering(features)
    labels = cluster_model.labels_
    print(cluster_model.labels_)