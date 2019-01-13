## Name: Deep convolutional autoencoder
## Description: Reads in the 32 by 32 image matrix
##              for each image and then trains a deep convolutional autoencoder
##              on the entire sketchy dataset using keras and tensorflow.
##              Then checks to see how well it can recreate images it has
##              never seen with some visualization
##              NOTE: This code uses the output from save32by32ForAll.py
## Author: Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build deep convolutional autoencoder
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

encoded_dim = 128

input_img = Input(shape=(32, 32, 1)) # define so rgb images can also be processed if necessary

x = Conv2D(32, 3, activation='relu', padding='same')(input_img) # convolutional layers
x = MaxPooling2D(padding='same')(x)
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(padding='same')(x)
x = Conv2D(8, 3, activation='relu', padding='same')(x)
x = Flatten()(x) # flatten convolutional layers and send to densely connected layers 
x = Dense(256, activation='relu')(x)
encoded = Dense(encoded_dim, activation='relu')(x)

# create encoder network
encoder = Model(input_img, encoded)

# define decoder model 
y = Dense(256, activation='relu')(encoded) # reflect flat, densely connected layers 
y = Dense(512, activation='relu')(y)
y = Reshape((8,8,8))(y)
y = Conv2D(16, 3, activation='relu', padding='same')(y) # reflect convolutional layers 
y = UpSampling2D()(y)
y = Conv2D(32, 1, activation='relu')(y)
y = UpSampling2D()(y)
decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(y) # output is 32 by 32 by 1 

# create autoencoder network
autoencoder = Model(input_img, decoded)

# create decoder network
# create separate decoder model by taking the last layers of the autoencoder network 
encoded_input = Input(shape=(encoded_dim,))
decoded_layer1 = autoencoder.layers[-8]
decoded_layer2 = autoencoder.layers[-7]
decoded_layer3 = autoencoder.layers[-6]
decoded_layer4 = autoencoder.layers[-5]
decoded_layer5 = autoencoder.layers[-4]
decoded_layer6 = autoencoder.layers[-3]
decoded_layer7 = autoencoder.layers[-2]
decoded_layer8 = autoencoder.layers[-1]
decoder = Model(encoded_input,
                decoded_layer8(decoded_layer7(decoded_layer6(decoded_layer5(decoded_layer4(decoded_layer3(decoded_layer2(decoded_layer1(encoded_input)))))))))

# define hyperparameters of autoencoder network
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
    numImgs = smallerImgs.shape[0]
    smallerImgs_reshape = np.zeros((numImgs, 32, 32, 1))
    counter = 0
    for img in smallerImgs:
        smallerImgs_reshape[counter, :, :, :] = np.reshape(img, (1, 32, 32, 1))
        counter += 1
    if folderCount == 0:
        array = smallerImgs_reshape
    else:
        array = np.append(array, smallerImgs_reshape, axis=0)
    folderCount += 1

np.random.shuffle(array)
nSamples = array.shape[0]
propTest = 0.1
nTrain = int((1-propTest)*nSamples)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

autoencoder.fit(array[0:nTrain,:,:,:], array[0:nTrain,:,:,:],
                epochs=30,
                batch_size=32,
                shuffle=True,
                validation_data=(array[nTrain:,:,:,:], array[nTrain:,:,:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Check encoded images
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# encode and decode some digits
encoded_imgs = encoder.predict(array[nTrain:,:,:,:])
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
startIndx = 4521 # start at some random point in the validation set 
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    i2 = nTrain + i + startIndx
    plt.imshow(array[i2,:,:,0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    i3 = i + startIndx
    plt.imshow(decoded_imgs[i3,:,:,0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
