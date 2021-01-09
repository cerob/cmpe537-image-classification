import numpy as np
from os import walk
import os
from pipeline import local_descriptors

class BagOfVW:

    def __init__(self):
        pass

    def compare_point_to_means(self, point, means):
        distances = []
        x = means - point
        distances = np.linalg.norm(x, axis=1)
        return distances

    def quantize(self, cluster_centers, datapoints):
        bag_of_vw=np.zeros(cluster_centers.shape[0])
        for i in range(len(datapoints)):
            dist = self.compare_point_to_means(datapoints[i], cluster_centers)
            # index = np.where(dist == np.amin(dist))
            index = np.unravel_index(np.argmin(dist), dist.shape)
            bag_of_vw[index[0].min()]=bag_of_vw[index[0].min()]+1

        max=np.max(bag_of_vw)
        bag_of_vw=bag_of_vw/max
        return bag_of_vw

    def load_folder(self,path):
        all_files = []
        classes = []
        docs = []
        for (dirpath, dirnames, filenames) in walk(path):
            for file in filenames:
                dirname = dirpath.replace(path + "\\", "")
                all_files.append(os.path.join(dirpath, file))
                classes.append(dirname)

        return all_files, classes

if __name__ == '__main__':
    bovw=BagOfVW()
    describers = local_descriptors.LocalDescriptors()
    train_images, train_classes = bovw.load_folder("C:/Users/yusuf/Desktop/MS/CMPE 537/Caltech20/training")
    test_images, test_classes = bovw.load_folder("C:/Users/yusuf/Desktop/MS/CMPE 537/Caltech20/testing")
    descriptions = []
    features=[]
    means=np.load("hier_kmeans.npy")
    for image in train_images:
        feature=bovw.quantize(means,describers.orb_descriptor(image,500))
        features.append(feature)

    print(features)
