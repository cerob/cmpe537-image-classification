import cv2
import numpy
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import pickle

def main():
    print("main")
    print("Os path: ", os.getcwd())
    currentPath = os.getcwd()
    currentPath = currentPath + "/Caltech20/training"
    CLUSTER_COUNT = 20 # which also indicates the quantization/histogram length
    kmeans = KMeans(n_clusters=CLUSTER_COUNT)

    categoryToFeaturesList, folderList, filesOfFolders = createCategoryForFeatureList(currentPath)

    categoryToFeatures = getCategoryToFeatures(folderList, currentPath, categoryToFeaturesList, CLUSTER_COUNT)

    allDescriptors = categoryToFeatures[0][0]
    for i in range(len(folderList)):
        for features in categoryToFeatures[i]:
            allDescriptors = numpy.vstack((allDescriptors, features))

    print("Descriptors shape: ", allDescriptors.shape)
    allDescriptors = allDescriptors.astype(float)

    numpy.save("AllTrainingDescriptors", allDescriptors)


    # load the model from disk after first run since it takes long time to compute

    #kmeans = pickle.load(open('kmeans_model.sav', 'rb'))
    #allDescriptors = numpy.load("AllTrainingDescriptors.npy")

    kmeans.fit(allDescriptors)
    #filename = 'kmeans_model.sav'
    #pickle.dump(kmeans, open(filename, 'wb'))


    trainingSet, trainingLabels = getTrainingSet(folderList, filesOfFolders, kmeans, categoryToFeatures, CLUSTER_COUNT)
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(trainingSet, trainingLabels)

    currentPath = os.getcwd()
    currentPath = currentPath + "/Caltech20/testing"

    categoryToFeaturesList, folderList, filesOfFolders = createCategoryForFeatureList(currentPath)

    categoryToFeatures = getCategoryToFeatures(folderList, currentPath, categoryToFeaturesList, CLUSTER_COUNT)

    testSet, testLabels = getTrainingSet(folderList, filesOfFolders, kmeans, categoryToFeatures, CLUSTER_COUNT)

    predictions = rf.predict(testSet)
    predictions = numpy.floor(predictions)

    numberOfPredictions = predictions.size

    classificationMatrix = predictions == testLabels
    correctlyClassified = numpy.sum(classificationMatrix)
    accuracy = (correctlyClassified / numberOfPredictions) * 100
    print("Accuracy: %", accuracy)


def createCategoryForFeatureList(currentPath):
    folderList = []
    categoryToFeaturesList = []  # actually it is a list to list, it will be used for training
    for f in listdir(currentPath):
        if f != ".DS_Store":
            folderList.append(f)
            categoryToFeaturesList.append([])

    filesOfFolders = []
    for folder in folderList:
        if folder != ".DS_Store":
            path = currentPath + "/" + folder
            for file in listdir(path):
                filesOfFolders.append(file)
    return categoryToFeaturesList, folderList, filesOfFolders


def getCategoryToFeatures(folderList, currentPath, categoryToFeatures, CLUSTER_COUNT):
    print("Classification starts...")
    #kmeans = KMeans(n_clusters=CLUSTER_COUNT)
    folderToImageCount = []
    for i in range(len(folderList)):
        imageCount = 0  # for a single category
        if folderList[i] != ".DS_Store":  # because of MacOS
            path = currentPath + "/" + folderList[i]
            for file in listdir(path):
                imageCount += 1
                filePath = path + "/" + file
                img = cv2.imread(filePath)
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(img, None)
                if len(kp1) > CLUSTER_COUNT:
                    categoryToFeatures[i].append(des1)
            print(folderList[i])
            folderToImageCount.append((folderList[i], imageCount))

    return categoryToFeatures


def getTrainingSet(folderList, fileList, kmeans, categoryToFeatures,CLUSTER_COUNT):
    allHistograms = numpy.zeros((len(fileList), CLUSTER_COUNT))
    allHistograms.astype(int)
    allLabels = numpy.zeros((len(fileList),))
    allLabels.astype(int)

    counter = 0  # holds total image number for training set
    for i in range(len(folderList)):  # for each category

        for features in categoryToFeatures[i]:  # for each image
            closestClusterList = kmeans.predict(features)
            allLabels[counter] = i  # which label/category
            for value in closestClusterList:
                allHistograms[counter][value] += 1
            counter += 1

    # if normalization is wanted on training set, then these lines can be used
    #denominatorForNormalization = numpy.max(allHistograms)
    #allHistograms = allHistograms / denominatorForNormalization
    return allHistograms, allLabels



if __name__ == '__main__':
    main()

