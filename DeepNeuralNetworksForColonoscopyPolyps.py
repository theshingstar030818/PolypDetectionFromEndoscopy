from __future__ import with_statement

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

from sklearn.metrics import roc_auc_score
from numpy.random import seed
# from tensorflow import set_random_seed
import tensorflow as tf
import time, os

class PlotLosses(keras.callbacks.Callback):
    # Show the live training and validation loss during the training
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        clear_output(wait=True)
        plt.plot(self.x, self.acc, label="ACC_Train")
        plt.plot(self.x, self.val_acc, label="ACC_Val")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.grid()
        # plt.ylim(0.0, 1.0) # limited Y values to 1.0!
        plt.show()

plot_losses = PlotLosses() # initialize the class

def CNN_3Conv_1FCv2(convs, dims, epochs, batch_size, drop_rate, input_shape):
    # CNN with 3 Conv + 1 FC
    # convs = no of filters
    # dims = dimension of convolution dims x dims
    # epochs = training epochs
    # batch_size = training batch size
    # drop_rate = drop rate (0, 1.0) <---- added
    # input_shape = shape of the inputs

    # for reproductibility
    seed(1)  # numpy seed
    # tf.set_random_seed(2) # tensorflow seed
    tf.random.set_seed(2)
    # Start the computational graph for our CNN
    model = Sequential()

    # Conv 1 filters
    model.add(Conv2D(convs, (dims, dims), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv 2 filters
    model.add(Conv2D(convs, (dims, dims), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv 3 filters
    model.add(Conv2D(convs * 2, (dims, dims)))  # second Conv has 2x filters!
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convert convolution to a fully connected layer
    model.add(Flatten())
    model.add(Dense(convs * 2))  # FC has 2x neurons!
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Aaugmentation configuration for training set
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90)

    # Augmentation configuration for testing set: only rescaling!
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[plot_losses],  # using the plotting function for train and validation loss
        verbose=0)

    # Evaluate final test loss and accuracy scores
    score_val = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size, workers=7)
    score_tr = model.evaluate_generator(train_generator, nb_train_samples // batch_size, workers=7)
    print('Train loss    :', score_tr[0])
    print('Train accuracy:', score_tr[1])
    print('Validation loss    :', score_val[0])
    print('Validation accuracy:', score_val[1])

    return model  # return the model!

# Dimensions of our images.
img_width, img_height = 150, 150

# Train & validation folders
train_data_dir = 'data_polyps/train'
validation_data_dir = 'data_polyps/validation'

# Folder to save the models
modelFolder = 'saved_models_CNN'

# Train parameters
nb_train_samples = 910
nb_validation_samples = 302

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print('--> Training best model for Small CNNs: Conv-Conv-Conv-FC ...')
model = CNN_3Conv_1FCv2(convs=64, dims=3, epochs=400, batch_size=64,
                        drop_rate=0.9, input_shape= input_shape)
print(model.summary())

# Save the weights and the full model
print('---> Save model ...')
model.save_weights(os.path.join(modelFolder,'model_best2_Conv-Conv-Conv-FC_weigths.h5'))
model.save(os.path.join(modelFolder,'model_best2_Conv-Conv-Conv-FC_full.h5'))
print('Done!')