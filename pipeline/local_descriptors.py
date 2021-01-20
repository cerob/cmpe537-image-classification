import cv2
from matplotlib import pyplot as plt
from os import walk
import os


class LocalDescriptors:

    def __init__(self):
        pass

    def orb_descriptor(self,file, nfeatures=500):
        img = cv2.imread(file)

        # Initiate STAR detector
        orb = cv2.ORB_create(nfeatures=nfeatures)

        # find the keypoints with ORB
        kp = orb.detect(img, None)

        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)

        if des is None:
            return []
        return des

    def hog_descriptor(self,file,nfeatures=500):
        pass



if __name__ == '__main__':
    pass