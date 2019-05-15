import glob
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
import keras

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build deep autoencoder
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

encoded_dim = 32
num_classes = 125

input_img = Input(shape=(32, 32, 1))

x = Conv2D(128, 3, activation='relu', padding='same')(input_img)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = Conv2D(8, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(padding='same')(x)
x = Conv2D(8, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(padding='same')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
encoded = Dense(encoded_dim, activation='relu')(x)

# create encoder network
encoder = Model(input_img, encoded)

y = Dense(256, activation='relu')(encoded)
y = Dense(512, activation='relu')(y)
y = Reshape((8,8,8))(y)
y = Conv2D(8, 3, activation='relu', padding='same')(y)
y = UpSampling2D()(y)
y = Conv2D(16, 3, activation='relu', padding='same')(y)
y = UpSampling2D()(y)
y = Conv2D(32, 3, activation='relu', padding='same')(y)
y = Conv2D(64, 3, activation='relu', padding='same')(y)
y = Conv2D(128, 3, activation='relu', padding='same')(y)
decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(y)

# create autoencoder network
autoencoder = Model(input_img, decoded)

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
classDict = {}
for folder in allFolders:
    print('Folder ' + str(folderCount+1) + ' of ' + str(len(allFolders)) + '.')
    smallerImgs = np.load(folder + '/32by32vecStack.npy')
    numImgs = smallerImgs.shape[0]
    smallerImgs_reshape = np.zeros((numImgs, 32, 32, 1))
    imgClass = folderCount*np.ones(smallerImgs.shape[0])
    className = folder.split('\\')[1]
    classDict[folderCount] = className
    counter = 0
    for img in smallerImgs:
        smallerImgs_reshape[counter, :, :, :] = np.reshape(img, (1, 32, 32, 1))
        counter += 1
    if folderCount == 0:
        array = smallerImgs_reshape
        labels = imgClass
    else:
        array = np.append(array, smallerImgs_reshape, axis=0)
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
## Train the autoencoder model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

autoencoder.fit(array[0:nTrain,:,:,:], array[0:nTrain,:,:,:],
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(array[nTrain:,:,:,:], array[nTrain:,:,:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build classifier network with encoder on the bottom
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x = encoder.output
# add a classification layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train for classification
model = Model(encoder.input, predictions)

# first we need to freeze all encoder layers so they
# are not trained by the classification process
for layer in encoder.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='Nadam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the classifier model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.fit(array[0:nTrain,:,:,:], labels_cat[0:nTrain,:],
                epochs=15,
                batch_size=32,
                shuffle=True,
                validation_data=(array[nTrain:,:,:,:], labels_cat[nTrain:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## unfreeze the densely connected layers of the encoder
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # print out the layer numbers and names to see
# for i, layer in enumerate(encoder.layers):
#    print(i, layer.name)

# unfreeze the last 3 densly connected layers of the encoder
for layer in encoder.layers[:10]:
   layer.trainable = False
for layer in encoder.layers[10:]:
   layer.trainable = True

# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Re-train the classifier model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.fit(array[0:nTrain,:,:,:], labels_cat[0:nTrain,:],
                epochs=15,
                batch_size=32,
                shuffle=True,
                validation_data=(array[nTrain:,:,:,:], labels_cat[nTrain:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Test classifier on dataset to check out true and false positives
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# create new arrays for easier book-keeping
X_test = array[nTrain:,:,:,:]
y_test = labels_cat[nTrain:,:]

# get predictions for validation data
classPreds = model.predict(X_test)

# go through predictions and actual labels to see which was right
realLabels = [] # grab the label
predLabels = [] # grab top label for each prediction
isRight = np.zeros(y_test.shape[0])
isRight_topThree = np.zeros(y_test.shape[0])
for i in range(0,y_test.shape[0]):
    # see if first prediction was right
    realIndx = np.argsort(y_test[i,:])[-1]
    predIndxs = np.argsort(classPreds[i,:])[-5:]
    predIndx = predIndxs[-1]
    if predIndx == realIndx:
        isRight[i] = 1
    # see if real label was in the top three predictions
    for pred in predIndxs:
        if pred == realIndx:
            isRight_topThree[i] = 1
    # grab real label for instance
    realLabels.append(classDict[realIndx])
    # grab three best predictions
    predLabels.append(classDict[predIndx])
