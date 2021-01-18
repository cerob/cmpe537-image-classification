import pickle
import numpy as np
from pipeline import local_descriptors
from sklearn import preprocessing
from pipeline import hier_kmeans
from pipeline import bag_of_vw
from pipeline import classifiers
from os import walk
import os
from imblearn import over_sampling


class BOVWPipeline:

    def __init__(self):
        pass

    def prepare_set(self, images, classes,x_name,y_name,balance,le):
        means=np.load("hier_kmeans.npy")
        descriptors=local_descriptors.LocalDescriptors()
        quantizer=bag_of_vw.BagOfVW()
        x = []
        y = []
        problematic_indices = []
        for i, image in enumerate(images):
            points = descriptors.orb_descriptor(image, 500)
            if len(points) == 0:
                problematic_indices.append(i)
            else:
                data = quantizer.quantize(means,points)
                x.append(data)


        y = le.transform(classes)
        y = np.delete(y, problematic_indices)
        X = np.stack(x, axis=0)
        if balance:
            X,y=self.balance_classes(X,y,100)
        np.save(x_name, X)
        np.save(y_name, y)
        return X, y

    def balance_classes(self, X, y,max):
        classes=np.unique(y)
        balanced_X=None
        balanced_y = None
        oversample = over_sampling.RandomOverSampler(sampling_strategy='minority')
        balanced_X,balanced_y = oversample.fit_resample(X, y)
        # for c in classes:
        #     if len(np.where(y == c)[0])>max:
        #         indices=np.where(y==c)
        #         print(indices)
        #         indices=(indices[0][0:max],)
        #         print(indices)
        #         if balanced_X is None:
        #             balanced_X=np.take(X,indices=indices,axis=0)
        #             balanced_y = np.take(y, indices=indices, axis=0)
        #         else:
        #             balanced_X=np.concatenate((balanced_X,np.take(X,indices=indices,axis=0)), axis=1)
        #             balanced_y=np.append(balanced_y,np.take(y,indices=indices,axis=0))
        #     else:
        #         indices=np.where(y==c)
        #         if balanced_X is None:
        #             balanced_X=np.take(X,indices=indices,axis=0)
        #             balanced_y = np.take(y, indices=indices, axis=0)
        #         else:
        #             balanced_X=np.concatenate((balanced_X,np.take(X,indices=indices,axis=0)), axis=1)
        #             balanced_y=np.append(balanced_y,np.take(y,indices=indices,axis=0))
        #
        # balanced_X = balanced_X.reshape((balanced_X.shape[1], balanced_X.shape[2]))
        return balanced_X,balanced_y

    def load_sets(self,x_name,y_name,balance):
        X=np.load(x_name)
        y=np.load(y_name)

        if balance:
            X,y=self.balance_classes(X,y,100)

        return X, y



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

    def pipeline(self):
        train_images, train_classes = self.load_folder("C:/Users/yusuf/Desktop/MS/CMPE 537/Caltech20/training")
        # le = preprocessing.LabelEncoder()
        # le.fit(train_classes)
        # pickle.dump(le, open("label_encoder.sav", 'wb'))
        le=pickle.load(open('label_encoder.sav', 'rb'))

        test_images, test_classes = self.load_folder("C:/Users/yusuf/Desktop/MS/CMPE 537/Caltech20/testing")
        # x_train, y_train = self.prepare_set(train_images, train_classes,"x_train_125_unb.npy","y_train_125_unb.npy",False,le)
        # x_test, y_test = self.prepare_set(test_images, test_classes,"x_test_125.npy","y_test_125.npy",False,le)
        x_train, y_train = self.load_sets("x_train_unb.npy","y_train_unb.npy",True)
        x_test, y_test = self.load_sets("x_test.npy","y_test.npy",False)
        # adaboost=classifiers.AdaboostClassifier()
        svm=classifiers.SVMClassifier()
        # mlp=classifiers.MLPClassifier()
        # adaboost.train(x_train,y_train)
        # adaboost.test(x_test,y_test)
        # svm.gridSearchTrain(x_train, y_train)
        # svm.test(x_train, y_train)
        # svm.test(x_test, y_test)
        svm.model_scores(x_test,y_test,le)
        # classes=le.inverse_transform(y_test)
        # print(classes)
        # mlp.gridSearchTrain(x_train, y_train)
        # mlp.test(x_train, y_train)
        # mlp.test(x_test, y_test)
        # mlp.model_scores(x_test, y_test)


if __name__ == '__main__':
    pipeline=BOVWPipeline()
    pipeline.pipeline()