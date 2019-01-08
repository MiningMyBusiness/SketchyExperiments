## Name: PNG to NPY
## Description: For the sketchy dataset, reads every image in the
##              sub folders and save the images both in vectorized
##              form and in matrix form for easier loading later on
## Author: Kiran Bhattacharyya
## Directions: To use this code. First download skecthy dataset from
##              http://sketchy.eye.gatech.edu/
##              Then place this code within the sketchy folder but
##              not in the subfolders. When you run this, it will save
##              two new files in every subfolder as within the directory
##              in the variable 'pngLoc' on line 20 below.


import glob
from scipy import misc
import numpy as np

# location of folders containing sketches
# that are 256 by 256 grayscale images
pngLoc = '256X256/sketch/tx_000100000000'

# grab folder names with sketches
allFolders = glob.glob(pngLoc + '/*')

# get names of folders for class names
classNames = [name.split('\\')[1] for name in allFolders]

# go through each folder and get names of images
# to create a list of lists
imgFilesPerFolder = []
for folder in allFolders:
    allImgs = glob.glob(folder + '/*.png')
    imgFilesPerFolder.append(allImgs)

# read each image in every folder to create
# 1. a stack of images that retains image shape 256 by 256 by num_of_images
# 2. reshape the image to a vector to have (256*256) 65536 by num_of_images
# 1. can be used to train convolutional neural nets
# 2. can be used to train basic MLPs and other autoregressive algorithms
folderNum = 0
for imgStack in imgFilesPerFolder:
    print('Processing folder number ' + str(folderNum+1) + ' of ' + str(len(allFolders)) + '.')
    num_of_images = len(imgStack)
    img_matStack = np.zeros((256, 256, num_of_images))
    img_vecStack = np.zeros((num_of_images, 65536))
    imgNum = 0
    for img in imgStack:
        thisImg = misc.imread(img)
        thisImg = np.mean(thisImg, axis=2) # average r,g,b channels
        thisImg = thisImg > 125 # thresholds image to make binary image
        thisImg_reshape = np.reshape(thisImg, 65536)
        img_matStack[:, :, imgNum] = thisImg
        img_vecStack[imgNum, :] = thisImg_reshape
        imgNum += 1
    np.save(allFolders[folderNum] + '/matStack.npy', img_matStack)
    np.save(allFolders[folderNum] + '/vecStack.npy', img_vecStack)
    print('        finished process.')
    folderNum += 1
