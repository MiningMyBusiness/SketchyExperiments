## Name: Save 32 by 32 image vector
## Description: Reads in the matrix representation of the images
##              for each class and resizes each image from 256 by 256
##              to 32 by 32 and then converts it to a vector with
##              1024 elements (32*32). A stack of these vectors (matrix)
##              is then saved in the class directory.
##              NOTE: This code uses the output from PNGtoNumpy.py
## Author: Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

# location of the data
dataLoc = '256X256/sketch/tx_000100000000'

allFolders = glob.glob(dataLoc + '/*') # get all folders with data

newSize = 32

def readImgAndResize(folderName, multi, newSize):
    imgStack_mat = np.load(folder + '/matStack.npy')
    num_of_images = imgStack_mat.shape[2]
    smallerImgs = np.zeros((num_of_images, newSize**2))
    for i in range(0, int(multi*num_of_images)):
        smallerImgs[i, :] = resizeAndVectorize(imgStack_mat[:, :, i], newSize)
    return smallerImgs

def resizeAndVectorize(thisImg, newSize):
    resized = cv2.resize(thisImg, (newSize, newSize),
                        interpolation=cv2.INTER_AREA)
    resized = np.reshape(resized, newSize**2)
    resized = resized - np.min(resized)
    resized = resized/np.max(resized)
    return resized


folderCount = 0
iter = 0
multi = 1.0
allData = []
for folder in allFolders:
    print('Folder ' + str(folderCount+1) + ' of ' + str(len(allFolders)) + '.')
    smallerImgs = readImgAndResize(folder, multi, newSize)
    np.save(folder + '/32by32vecStack.npy', smallerImgs)
    folderCount += 1
