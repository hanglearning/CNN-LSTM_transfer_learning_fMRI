# Syed Ali Asif developed the CNN-LSTM model
# Hang Chen evaluated the transfer learning performance on this model
# Nov 2019, healthy lAIfe lab at the University of Delaware

from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
        MaxPooling2D)
import keras.backend as K
from collections import deque
import sys


import keras
from keras.layers import Input ,Dense, Dropout, Activation, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model

import os

'''
argvs:

1 ~ 3: must have
1 - dataset to train on
2 - validation_split ratio
3 - how many epochs to train

4 ~ 7: optional, but must have 4 to have 5~7
4 - input any letter to do transfer learning. If not specified, will do normal weights training on 1.
5 - which data set to transfer the weights from
6 - how many epochs weights to be loaded from the transferred
7 - how many last few layers to be retrained
8 - any letter to store the transfer learning weight
'''

TRANSFER_LEARNING = False
TRAIN_LAYERS = 0
TRANSFER_FROM = ''
TRANSFER_STORE_WEIGHTS = False

try:
    if sys.argv[4]:
        TRANSFER_LEARNING = True
        TRANSFER_FROM = sys.argv[5]
        TRAIN_LAYERS = int(sys.argv[7])
        if sys.argv[4]:
            TRANSFER_STORE_WEIGHTS = True
except:
    pass

# nb_classes = 1;
nb_epoch = int(sys.argv[3]);
validation_split = float(sys.argv[2]);
batch_size=1;

#my data loading
import numpy as np
#import nibabel as nib

data_set_seq = sys.argv[1] # 001, 002, 003
dataset_path = f"/pylon5/cc5piep/chenhang/ds/{data_set_seq}/calorie_data_{data_set_seq}.npy"

if TRANSFER_LEARNING:
    print(f"Transfer learning from dataset {TRANSFER_FROM} on dataset {data_set_seq} with {nb_epoch} epochs only on the last {TRAIN_LAYERS} layers, with validation split ratio {validation_split}.")
else:
    print(f"Training on dataset {data_set_seq} with {nb_epoch} epochs, with validation split ratio {validation_split}.")

label_file = f"/pylon5/cc5piep/chenhang/ds/{data_set_seq}/{data_set_seq}labels.csv"
raw_data = open(label_file, 'rt')
label = np.loadtxt(raw_data, delimiter=",")
print(label.shape)
print(label)

final_data = np.load(dataset_path)
print("Dataset Shape", final_data.shape)
#print(final_data)


X_train = final_data[:,:,:,:,:]
print(X_train.shape)
y_train = label[:]

# print(final_data[41:,141:,:,:,:])

shape = X_train.shape[1:]
print(shape)



model = Sequential()

model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
        activation='relu', padding='same'), input_shape=shape))
model.add(TimeDistributed(Conv2D(32, (3,3),
        kernel_initializer="he_normal", activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last')))


model.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last')))

model.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last')))

model.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2),data_format='channels_last')))
                
model.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last')))

model.add(TimeDistributed(Flatten()))

model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=False, dropout=0.5))
model.add(Dense(1, activation='linear'))

import tensorflow as tf
# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

# set trainable layers if do transfer learning
if TRANSFER_LEARNING:
    for layer in model.layers[:-TRAIN_LAYERS]:
        layer.trainable = False

model.compile(loss='mean_squared_error',
                            optimizer='RMSprop',
                            metrics=['mae','mse'])

checkpoint_path = ''
checkpoint_dir = ''

if TRANSFER_LEARNING:
    # Load weights
    weights_path = f"/pylon5/cc5piep/chenhang/weights/{TRANSFER_FROM}/cp_{TRANSFER_FROM}_{sys.argv[6]}.ckpt"
    model.load_weights(weights_path)

    if not TRANSFER_STORE_WEIGHTS:

        model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs =nb_epoch,
                validation_split = validation_split,
                #validation_data=(X_test, y_test)
                )
        del final_data
        exit(0)
    else:
        checkpoint_path = f"/pylon5/cc5piep/chenhang/weights/cp_transfer_from_{TRANSFER_FROM}_{sys.argv[6]}_training_on_{data_set_seq}_{nb_epoch}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
else:
    # Store weights
    checkpoint_path = f"/pylon5/cc5piep/chenhang/weights/{data_set_seq}/cp_{data_set_seq}_{nb_epoch}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.fit(X_train, y_train,
        batch_size=batch_size,
        epochs =nb_epoch,
        validation_split = validation_split,
        callbacks = [cp_callback]
        #validation_data=(X_test, y_test)
        )

del final_data

#print('predicted label: ', model.predict(X_test), 'actual label:', y_test)
#print('predicted label: ', model.predict(X_test1), 'actual label:', y_test1)

#pred_train= model.predict(X_train)
#print(np.sqrt(mean_squared_error(y_train,pred_train)))

#pred= model.predict(X_test)
#print(np.sqrt(mean_squared_error(y_test,pred))) 