from matplotlib import pyplot as plt
import numpy as np
from pipeline import local_descriptors
from os import walk
import os

class HierarchicalKMeans:

    def kmeans(self, K, iterations, datapoints):
        means=self.get_means_by_random(K,datapoints[0].size)
        clusters = [[] for i in range(K)]
        cluster_indices = [[] for i in range(K)]
        labels = []
        # old_labels=[]
        for iter in range(iterations):
            clusters = [[] for i in range(K)]
            labels = []
            for i in range(len(datapoints)):
                dist = self.compare_point_to_means(datapoints[i], means)
                # index = np.where(dist == np.amin(dist))
                index = np.unravel_index(np.argmin(dist), dist.shape)
                clusters[index[0].min()].append(datapoints[i])
                labels.append(index[0].min())
            means = self.reset_means(clusters,datapoints[0].size)
            print(str(iter))
        # means = means.astype(int)
        return labels, means, clusters

    def compare_point_to_means(self, point, means):
        distances = []
        x = means - point
        distances = np.linalg.norm(x, axis=1)
        return distances

    def reset_means(self, clusters, features):
        new_means = []
        for i in range(len(clusters)):
            if len(clusters[i]) > 0:
                new_means.append(np.mean(clusters[i], axis=0))
            else:
                new_means.append(np.random.randint(0, 256, features))

        new_means = np.array(new_means)
        # print(new_means)
        return new_means

    def get_means_by_random(self,K,features):
        means = []
        for i in range(K):
            point = np.random.uniform(0, 256, features)
            means.append(point)
        means = np.array(means)
        # print(means)
        return means

    def hier_k_means(self, K, layers, iterations, datapoints):#can be better implemented
        means=[]
        labels=[]
        data=datapoints
        data2=[]
        # l,m,clusters=self.kmeans(K, iterations, data)
        #
        # for cluster in clusters:
        #     l1,m1,cs=self.kmeans(K, iterations, cluster)


        for i in range(layers):
            if i==0:
                print("LAYER "+str(i)+" ITERATION 1")
                l,m,cs=self.kmeans(K,iterations,data)
                data=cs
            if i==1:
                for ind,c in enumerate(data):
                    print("LAYER "+str(i)+" ITERATION "+str(ind))
                    l,m,cs=self.kmeans(K,iterations,c)
                    data2.extend(cs)
            if i==2:
                for ind,c in enumerate(data2):
                    print("LAYER "+str(i)+" ITERATION "+str(ind))
                    l,m,cs=self.kmeans(K,iterations,c)
                    means.extend(m)
                    labels.extend(l)
        return means

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
    kmeans=HierarchicalKMeans()
    describers=local_descriptors.LocalDescriptors()
    train_images, train_classes = kmeans.load_folder("C:/Users/yusuf/Desktop/MS/CMPE 537/Caltech20/training")
    test_images, test_classes = kmeans.load_folder("C:/Users/yusuf/Desktop/MS/CMPE 537/Caltech20/testing")
    descriptions = []
    for image in train_images:
        descriptions.extend(describers.orb_descriptor(image, 500))

    print(len(descriptions))
    means=kmeans.hier_k_means(5,3,10,descriptions)
    np.save("hier_kmeans.npy", means)
    print(means)
