## Name: Sonify N-gram Drawing 2
## Description:  Uses learned n-grams from all of the sketches
##              in the sketchy dataset to make an n-gram image
##              this first makes a portion of the image as a
##              vector until it can start making a 2D ngram
##              where the probability of the next pixel depends
##              on the the preceeding n pixels in the row and
##              the column. Then it turns this
##              image into a sound wave by treating the y-axis of the
##              image as frequencies and the x-axis as time.
##              NOTE: This code needs the output from the code
##              drawingNGram.py found in the Pixel N-gram directory
## Author: Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

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

## make pixel ngram into music
sampRate = 44100 # common sampling rate of sound
timePoints = 256
secPerPoint = 0.1 # seconds per pixel
dataPoints = np.round(sampRate*timePoints*secPerPoint)
mySong = np.zeros((2, int(dataPoints))) # numchannels by numsamples (stereo sound)
freqs = np.logspace(1.0, 4.3, 256) # frequency representation of each pixel
stereoSteps = 1.0/256 # pan each frequency to a specific stereo sound location
stereoVal = np.arange(0., 1., stereoSteps)

# function that takes a row in the image and produces
# a sound wave based on the frequency that row represents
def getSoundRow(dataPoints, sampRate, imageRow, freq):
    myTime = (1.0/sampRate)*np.arange(0, dataPoints)
    mySine = np.sin(myTime*freq*np.pi)
    sineIncr = int(dataPoints/np.max(imageRow.shape))
    for i,val in enumerate(imageRow):
        if val > 0:
            indxStart = i*sineIncr
            indxLast = i*sineIncr + sineIncr
            mySine[indxStart:indxLast] = 0
    return mySine

## go through each row and populate the song variable
for rowNum, imageRow in enumerate(fullImg):
    print('Processing row number ' + str(rowNum+1) + '/' + str(len(fullImg)))
    soundWave = getSoundRow(dataPoints, sampRate,
                            imageRow, freqs[rowNum])
    mySong[0,:] += stereoVal[rowNum]*soundWave
    mySong[1,:] += (1 - stereoVal[rowNum])*soundWave

multi = (2**15)/np.max(mySong)
mySong = np.int16(mySong*multi)

n_gramSongs = glob.glob(str(n_gram) + '_Song2D*')
songNum = len(n_gramSongs) + 1
scipy.io.wavfile.write(str(n_gram) + '_Song2D_left_' + str(songNum) + '.wav',
                        sampRate, mySong[0,:])
scipy.io.wavfile.write(str(n_gram) + '_Song2D_right_' + str(songNum) + '.wav',
                        sampRate, mySong[1,:])
