import os
import argparse
import timeit
import cv2
import numpy as np
from scipy.io import loadmat,savemat
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger,ModelCheckpoint
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dense, Reshape, Flatten
from tf_capsulelayers import CapsuleLayer,PrimaryCap
from tensorflow.keras.regularizers import l1,l2,l1_l2
import wandb

K.set_image_data_format('channels_last')
print('Initializing pre-trained model..')
pre_trained_model = VGG19()

NA = 'not available'

# PARAMETERS
RUN = 1
ROUTINGS = 3
N_CLASS = 10	# complexity
OPTIMIZER = 'adam'
LOSS = 'mse'
EPOCHS = 2
BATCH_SIZE = 128
DATASET = 'ntire'
DATA_PATH = '../data/train_128_128.npz'
PRETRAINED_MODEL_PATH = 'pretrained_model.h5'

# PATHS
RUN_PATH = os.path.join('runs', str(RUN))
MODEL_PATH = os.path.join(RUN_PATH, 'model_'+str(EPOCHS)+'.h5')
MODEL_SUMMARY_PATH = os.path.join(RUN_PATH, 'model_summary.txt')
OUT_PATH = os.path.join(RUN_PATH, 'out')
LOG_PATH = os.path.join(RUN_PATH, 'log.csv')

def load_ntire():
    data = np.load(DATA_PATH)
    x_gray = data['arr_0']
    x_color = data['arr_1']
    x_gray = x_gray.astype('float32') / 255.
    x_color = x_color.astype('float32') / 255.
    return (x_gray,x_color)

def build_model(input_shape):
    # encoder
    x = Input(shape=input_shape)

    conv1 = Conv2D(filters=64, kernel_size=3, padding='same', trainable=True, name='conv1')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=64, kernel_size=3, padding='same', trainable=True, name='conv2')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=N_CLASS, dim_capsule=16, routings=ROUTINGS,
                             name='digitcaps')(primarycaps)

    digitcaps = Flatten()(digitcaps)
    decoder = Dense(512, activation='relu', input_dim=16*N_CLASS)(digitcaps)
    decoder = Dense(1024, activation='relu')(decoder)
    decoder = Dense(np.prod(input_shape), activation='sigmoid')(decoder)
    decoder = Reshape(input_shape)(decoder)

    model = Model(x, decoder)

    # transfer weights from first 2 layers of VGG-19
    weights = pre_trained_model.layers[1].get_weights()[0][:,:,:,:]
    #weights = np.reshape(weights, (3,3,1,64))
    bias = pre_trained_model.layers[1].get_weights()[1]
    model.layers[1].set_weights([weights, bias])
    weights = pre_trained_model.layers[2].get_weights()[0]
    bias = pre_trained_model.layers[2].get_weights()[1]
    model.layers[4].set_weights([weights, bias])

    return model

build_model((64,64,3))