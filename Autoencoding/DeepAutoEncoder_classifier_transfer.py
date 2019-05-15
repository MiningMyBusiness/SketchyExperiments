## Name: Deep autoencoder
## Description: Reads in the 1024 element (32 by 32) vector
##              for each image and then trains a deep autoencoder
##              on the entire sketchy dataset using keras and tensorflow.
##              Then checks to see how well it can recreate images it has
##              never seen with some visualization
##              NOTE: This code uses the output from save32by32ForAll.py
## Author: Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout
from keras.models import Model
import keras

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build deep autoencoder
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# encoded dimension
encoded_dim = 256

# number of classes
num_classes = 125

# input placeholder
input_img = Input(shape=(32**2,))
# 'encoder' model
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(encoded_dim, activation='relu')(encoded)

# 'decoder' model
decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(32**2, activation='sigmoid')(decoded)

# create seperate encoder
encoder = Model(input_img, encoded)

# full autoeconder model
autoencoder = Model(input_img, decoded)

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
    imgClass = folderCount*np.ones(smallerImgs.shape[0])
    if folderCount == 0:
        array = smallerImgs
        labels = imgClass
    else:
        array = np.append(array, smallerImgs, axis=0)
        labels = np.append(labels, imgClass)
    folderCount += 1

# shuffle array and labels
rng_state = np.random.get_state()
np.random.shuffle(array)
np.random.set_state(rng_state)
np.random.shuffle(labels)
labels_cat = keras.utils.to_categorical(labels, num_classes)
nSamples = array.shape[0]
propTest = 0.1
nTrain = int((1-propTest)*nSamples)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the autencoder model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

autoencoder.fit(array[0:nTrain,:], array[0:nTrain,:],
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(array[nTrain:,:], array[nTrain:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build classifier network with encoder on the bottom
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x = encoder.output
# add a classification layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train for classification
model = Model(encoder.input, predictions)

# first we need to freeze all encoder layers so they
# are not trained by the classification process
for layer in encoder.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the classifier model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.fit(array[0:nTrain,:], labels_cat[0:nTrain,:],
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(array[nTrain:,:], labels_cat[nTrain:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## unfreeze the densely connected layers of the encoder
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # print out the layer numbers and names to see
# for i, layer in enumerate(encoder.layers):
#    print(i, layer.name)

# unfreeze the last 2 densly connected layers of the encoder
for layer in encoder.layers[:3]:
   layer.trainable = False
for layer in encoder.layers[3:]:
   layer.trainable = True

# we use Adagrad
model.compile(optimizer='Adagrad',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Re-train the classifier model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.fit(array[0:nTrain,:], labels_cat[0:nTrain,:],
                epochs=15,
                batch_size=32,
                shuffle=True,
                validation_data=(array[nTrain:,:], labels_cat[nTrain:,:]))
