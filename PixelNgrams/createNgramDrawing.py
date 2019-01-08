## Name: Create N-gram Drawing
## Description: Uses learned n-grams from all of the sketches
##              in the sketchy dataset to make an n-gram image
##              this first makes a vector that is 65536 (256 by 256)
##              elements long and then rearranges it.
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

def createImage(myDict, n_gram):
    imgLength = 256**2
    fullImage = np.zeros(imgLength)
    firstGram = np.ones(n_gram)
    iter = n_gram
    fullImage[iter-n_gram:iter] = firstGram
    nextProb = myDict[tuple(firstGram)]
    if nextProb >= np.random.uniform(1, 0, 1):
        fullImage[iter] = 1
    iter += 1
    while iter < imgLength - n_gram:
        lastGram = fullImage[iter-n_gram:iter]
        nextProb = myDict[tuple(lastGram)]
        if nextProb >= np.random.uniform(1, 0, 1):
            fullImage[iter] = 1
        iter += 1
    return fullImage

fullImg_vec = createImage(myDict, n_gram)
fullImg = np.reshape(fullImg_vec, (256, 256))
plt.imshow(fullImg)
plt.show()
