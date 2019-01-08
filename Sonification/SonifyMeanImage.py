## Name: Sonify Mean Image
## Description: Computes the mean image from each category (airplane,
##              bell, etc.) and sonifies the mean image by using the
##              y-axis of the image to denote frequencies, the
##              x-axis to denote time and the value of each pixel
##              to denote the amplitude.
##              NOTE: This code needs the output from the code
##              PNGtoNumpy.py found in the Pixel N-gram directory
## Author: Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

# location of the data
dataLoc = '256X256/sketch/tx_000100000000'

allFolders = glob.glob(dataLoc + '/*') # get all folders with data

def getMeanImg(folderName):
    imgStack_mat = np.load(folderName + '/matStack.npy')
    meanImg = np.mean(imgStack_mat, axis=2)
    numImgs = imgStack_mat.shape[2]
    return meanImg, numImgs

## make pixel ngram into music
sampRate = 44100 # common sampling rate of sound
timePoints = 256
secPerPoint = 0.1
dataPoints = np.round(sampRate*timePoints*secPerPoint)
freqs = np.logspace(1.0, 4.3, 256)
stereoSteps = 1.0/256
stereoVal = np.arange(0., 1., stereoSteps)

# function that takes data and produces appropriate file
def getSoundRow(dataPoints, sampRate, imageRow, freq):
    myTime = (1.0/sampRate)*np.arange(0, dataPoints)
    mySine = np.sin(myTime*freq*np.pi)
    sineIncr = int(dataPoints/np.max(imageRow.shape))
    for i,val in enumerate(imageRow):
        val = 1.0 - val # invert value
        indxStart = i*sineIncr
        indxLast = i*sineIncr + sineIncr
        mySine[indxStart:indxLast] = val*mySine[indxStart:indxLast]
    return mySine

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

folderCount = 0
iter = 0
for folder in allFolders:
    print('Folder ' + str(folderCount+1) + ' of ' + str(len(allFolders)) + '.')
    mySong = np.zeros((2, int(dataPoints))) # numchannels by numsamples (stereo sound)
    meanImg, numImgs = getMeanImg(folder)
    for rowNum, imgRow in enumerate(meanImg):
        thisRow = getSoundRow(dataPoints, sampRate, imgRow, freqs[rowNum])
        mySong[0,:] += stereoVal[rowNum]*thisRow
        mySong[1,:] += (1.0 - stereoVal[rowNum])*thisRow

    multi = (2**15)/np.max(mySong)
    mySong = np.int16(mySong*multi)

    scipy.io.wavfile.write(folder + '/meanImgSong_left.wav',
                            sampRate, mySong[0,:])
    scipy.io.wavfile.write(folder + '/meanImgSong_right.wav',
                            sampRate, mySong[1,:])

    plt.imshow(meanImg)
    plt.savefig(folder + '/meanImg_from' + str(int(numImgs)) + 'images.png')   # save the figure to file
    plt.close()    # close the figure

    folderCount += 1
