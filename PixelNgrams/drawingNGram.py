## Name: Drawing N-gram
## Description: Uses the output of the PNGtoNumpy.py code to learn
##              pixel n-grams from all the sketches in the dataset
##              NOTE: you must first run the PNGtoNumpy.py code so
##              that all of the correct .npy files exist in the right
##              locations
## Author: Kiran Bhattacharyya
## Directions: To run this code please first download the sketchy
##             dataset, then run the PNGtoNumpy.py code. Then place
##             this code file in the sketchy directory and run it as well


import glob
import numpy as np

# location of the data
dataLoc = '256X256/sketch/tx_000100000000'

## first find all of the possible n-combinations present in
## the data set

n_gram = 12 # define n-gram size

allFolders = glob.glob(dataLoc + '/*') # get all folders with data

allChars_indx = {}
allChars_indx_rev = {}
allChars_data = {}
folderCount = 0
iter = 0
for folder in allFolders:
    print('Folder ' + str(folderCount) + ' of ' + str(len(allFolders)) + '.')
    imgStack_vec = np.load(folder + '/vecStack.npy')
    imgCount = 0
    for img in imgStack_vec:
        print('    Image ' + str(imgCount) + ' of ' + str(len(imgStack_vec)) + '.')
        for i in range(0, len(img) - n_gram):
            thisGram = img[i:i+n_gram]
            thisGram = tuple(thisGram)
            nextVal = img[i+n_gram]
            if thisGram not in allChars_indx:
                allChars_indx[thisGram] = iter
                allChars_indx_rev[iter] = thisGram
                allChars_data[thisGram] = [int(nextVal), 1]
                iter += 1
            else:
                allChars_data[thisGram][1] += 1
                if nextVal > 0:
                    allChars_data[thisGram][0] += 1
        imgCount += 1
        if imgCount > (0.1*len(imgStack_vec)): # you've read 10% of the images
            break
    folderCount += 1

allKeys = allChars_indx.keys()
allKeys_arr = np.zeros((len(allKeys), n_gram))
oneProb_arr = np.zeros(len(allKeys))
iter = 0
for key in allKeys:
    allKeys_arr[iter, :] = np.array(key)
    dataList = allChars_data[key]
    oneProb_arr[iter] = float(dataList[0])/float(dataList[1])
    iter += 1

np.save(str(n_gram) + '_gramKeys.npy', allKeys_arr)
np.save(str(n_gram) + '_probNextOne.npy', oneProb_arr)
