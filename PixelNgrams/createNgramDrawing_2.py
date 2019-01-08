## Name: Create N-gram Drawing 2
## Description: Uses learned n-grams from all of the sketches
##              in the sketchy dataset to make an n-gram image
##              this first makes a portion of the image as a
##              vector until it can start making a 2D ngram
##              where the probability of the next pixel depends
##              on the the preceeding n pixels in the row and
##              the column.
##              NOTE: This code needs to output from the code
##              drawingNGram.py
## Author: Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt

n_gram = 12
allKeys = np.load(str(n_gram) + '_gramKeys.npy')
probOne = np.load(str(n_gram) + '_probNextOne.npy')

# create dictionary
myDict = {}
for i, key in enumerate(allKeys):
    myDict[tuple(key)] = probOne[i]

def createImage(myDict, n_gram, allKeys):
    partRows = n_gram
    partImg = partRows*256
    partVec = np.zeros(partImg)
    firstGram = np.ones(n_gram)
    iter = n_gram
    partVec[iter-n_gram:iter] = firstGram
    nextProb = myDict[tuple(firstGram)]
    if nextProb >= np.random.uniform(1, 0, 1):
        partVec[iter] = 1
    iter += 1
    while iter < partImg - n_gram:
        lastGram = partVec[iter-n_gram:iter]
        nextProb = myDict[tuple(lastGram)]
        if nextProb >= np.random.uniform(1, 0, 1):
            partVec[iter] = 1
        iter += 1
    # rearrange part vector into partial image
    fullImage = np.zeros((256, 256))
    fullImage[0:partRows, :] = np.reshape(partVec, (partRows, 256))
    for i in range(0, 256 - partRows):
        i += partRows
        print(i)
        for j in range(0, 256):
            if j < n_gram:
                lastGram = np.transpose(fullImage[i-n_gram:i, j])
                try:
                    nextProb = myDict[tuple(lastGram)]
                except KeyError:
                    lastGram = gramDiff(allKeys, lastGram)
                    nextProb = myDict[tuple(lastGram)]
                if nextProb >= np.random.uniform(1, 0, 1):
                    fullImage[i,j] = 1
            else:
                lastGram_1 = np.transpose(fullImage[i-n_gram:i, j])
                try:
                    nextProb_1 = myDict[tuple(lastGram_1)]
                except KeyError:
                    lastGram_1 = gramDiff(allKeys, lastGram_1)
                    nextProb_1 = myDict[tuple(lastGram_1)]
                lastGram_2 = fullImage[i-1,j-n_gram:j]
                try:
                    nextProb_2 = myDict[tuple(lastGram_2)]
                except KeyError:
                    lastGram_2 = gramDiff(allKeys, lastGram_2)
                    nextProb_2 = myDict[tuple(lastGram_2)]
                nextProb = (nextProb_1 + nextProb_2)/2.0
                if nextProb >= np.random.uniform(1, 0, 1):
                    fullImage[i,j] = 1
    return fullImage

def gramDiff(allKeys, thisGram):
    thisDiff = np.sum(np.absolute(allKeys - thisGram), axis=1)
    minIndx = np.argmin(thisDiff)
    updateGram = allKeys[minIndx, :]
    return updateGram

fullImg = createImage(myDict, n_gram, allKeys)
plt.imshow(fullImg)
plt.show()
