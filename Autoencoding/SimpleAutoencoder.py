## Name: Simple autoencoder
## Description: Reads in the 1024 element (32 by 32) vector
##              for each image and then trains a simple autoencoder
##              on the entire sketchy dataset using keras and tensorflow.
##              Then checks to see how well it can recreate images it has
##              never seen with some visualization
##              NOTE: This code uses the output from save32by32ForAll.py
## Author: Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build simple autoencoder
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# encoded representation
# encoded_dim will be n of an n by n image
encoded_dim = 102

# input placeholder
input_img = Input(shape=(32**2,))
# 'encoder' model
encoded = Dense(encoded_dim, activation='relu')(input_img)
# 'decoder' model
decoded = Dense(32**2, activation='sigmoid')(encoded)

# full autoeconder model
autoencoder = Model(input_img, decoded)

# create seperate encoder
encoder = Model(input_img, encoded)

# create separate decoder model
encoded_input = Input(shape=(encoded_dim,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoded_layer(encoded_input))

# configure model to be trained
# per-pixel binary crossentropy loss
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Load in and process data
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# location of the data
dataLoc = '256X256/sketch/tx_000100000000'

allFolders = glob.glob(dataLoc + '/*') # get all folders with data

folderCount = 0
iter = 0
allData = []
for folder in allFolders:
    print('Folder ' + str(folderCount+1) + ' of ' + str(len(allFolders)) + '.')
    smallerImgs = np.load(folder + '/32by32vecStack.npy')
    if folderCount == 0:
        array = smallerImgs
    else:
        array = np.append(array, smallerImgs, axis=0)
    folderCount += 1

nSamples = array.shape[0]
propTest = 0.1
nTrain = int((1-propTest)*nSamples)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

autoencoder.fit(array[0:nTrain,:], array[0:nTrain,:],
                epochs=30,
                batch_size=32,
                shuffle=True,
                validation_data=(array[nTrain:,:], array[nTrain:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Check encoded images
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# encode and decode some digits
encoded_imgs = encoder.predict(array[nTrain:,:])
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    i2 = nTrain + i
    plt.imshow(array[i2].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
