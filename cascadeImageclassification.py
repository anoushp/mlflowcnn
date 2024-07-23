#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:56:35 2022

@author: apoghosyan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this code two custom CNNs are created on lego bricks image dataset and the models are run on mlflow. Parameters and artifacts are logged 
accordingly.
Created on Tue Sep 21 14:31:47 2021

@author: apoghosyan
"""

import csv

import numpy as np


import pandas as pd


import mlflow
import mlflow.sklearn

# Import Tensorflow
import tensorflow as tf

# Import needed tools.
import os
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy import stats
import pandas as pd
# Viewing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import perf_counter
# Import Keras
import tensorflow.keras
from tensorflow.keras.layers import Dense,Flatten, Dropout, Lambda
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D, Conv2D, Activation
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from mlflow.keras import log_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, DepthwiseConv2D, LeakyReLU, Add, GlobalMaxPooling2D
from keras.models import Model
from skimage.transform import resize
from PIL import Image, ImageOps
import tensorflow as tf

import pickle
import warnings
import hydra
from hydra import utils

import os
import json
import keras


import os
#from evidently.model_profile import Profile
#from evidently.profile_sections import DataDriftProfileSection

#################################
# Create model
#################################

def get_optimizer(optimizer, learning_rate = 0.001):
    if optimizer == 'adam':
        return keras.optimizers.Adam(learning_rate)
    elif optimizer == 'sgd':
        return tensorflow.keras.optimizers.SGD(lr = learning_rate, momentum = 0.99) 
    elif optimizer == 'adadelta':
        return tensorflow.keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)

def read_file(name):
    print(utils.to_absolute_path(name))
    
    f = pd.read_csv(utils.to_absolute_path(name),index_col=0)
    return f

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
  
 #resize images so they all are the same size
def resize_pad(img):
    old_size = img.shape[:2]
    ratio = 200. / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)
    
    delta_w = 200 - new_size[1]
    delta_h = 200 - new_size[0]
    padding = ((delta_h//2, delta_h-(delta_h//2)), (delta_w//2, delta_w-(delta_w//2)), (0, 0))
    
    img = np.pad(img, padding, 'edge')
    
    return img

def preprocessing_train(x):
    x = resize_pad(x)
    return x

def preprocessing_val(x):
    x = resize_pad(x)
    return x
def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history['loss'], label='Train loss')
    ax[0].plot(history.epoch, history.history['val_loss'], label='Validation loss')
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history['acc'], label='Train acc')
    ax[1].plot(history.epoch, history.history['val_acc'], label='Validation acc')
    ax[0].legend()
    ax[1].legend()
    return fig

#################################
# Model Building
#################################
def get_model(model):
# Load the pretained model
    kwargs =    {'input_shape':(224, 224, 3),
                'include_top':False,
                'weights':'imagenet',
                'pooling':'avg'}
    
    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False
    
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(20, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    return model
#creating customised CNN
def test_model2(opt, input_shape, dropout = 0.0):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8, (3, 3), input_shape = input_shape))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(tf.keras.layers.Dropout(dropout))
    
    model.add(tf.keras.layers.Dense(20, activation = 'softmax'))
    return model
#creating customised CNN
def test_model(opt, input_shape, dropout = 0.0):
    inputs = Input(shape=(200, 200, 3))
    net = Conv2D(filters=64, kernel_size=3, padding='same')(inputs)
    net = LeakyReLU()(net)
    net = MaxPooling2D()(net)
    
    net = Conv2D(filters=64, kernel_size=3, padding='same')(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D()(net)
    
    net = Conv2D(filters=64, kernel_size=3, padding='same')(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D()(net)
    
    shortcut = net
    
    net = DepthwiseConv2D(kernel_size=3, padding='same')(net)
    net = LeakyReLU()(net)
    
    net = Conv2D(filters=64, kernel_size=1, padding='same')(net)
    net = LeakyReLU()(net)
    
    net = DepthwiseConv2D(kernel_size=3, padding='same')(net)
    net = LeakyReLU()(net)
    
    net = Conv2D(filters=64, kernel_size=1, padding='same')(net)
    net = LeakyReLU()(net)
    
    net = Add()([shortcut, net])
    
    net = Conv2D(filters=64, kernel_size=3, padding='same')(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D()(net)
    
    net = Conv2D(filters=64, kernel_size=3, padding='same')(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D()(net)
    
    net = DepthwiseConv2D(kernel_size=3, padding='same')(net)
    net = LeakyReLU()(net)
    
    net = Conv2D(filters=128, kernel_size=1, padding='same')(net)
    net = LeakyReLU()(net)
    
    net = Flatten()(net)
    
    net = Dense(128, activation='relu')(net)
    
    net = Dense(64, activation='relu')(net)
    
    outputs = Dense(20, activation='softmax')(net)

    model = Model(inputs=inputs, outputs=outputs)
    
    return model    

#hydra config directory should be amended accordingly. configs is uploaded to gitlab with all the codes
@hydra.main(config_path='/gpfs/cfms/home/apoghosyan/tabular-playground-series-nov-2021/mlflow-legocnn/configs',
            config_name='hyperparameters')
def main(config):
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    input_shape             = (200, 200, 3) # This is the shape of the image width, length, colors
    image_size              = (input_shape[0], input_shape[1]) # DOH! image_size is (height, width)
    train_test_ratio        = 0.3
    zoom_range              = 0.3
    shear_range             = 0.2
    print(config.model.save_dir)
    print(config.base_path)
    #################################
    # Create needed dirs
    #################################
    make_dir(config.model.save_dir)

    
    
    train_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_train,
    rescale=1./255,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    validation_split=0.1
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_val,
        rescale=1./255,
        validation_split=0.1
    )

    train_generator = train_datagen.flow_from_directory(
    config.base_path,
    color_mode='rgb',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=0
    )

    validation_generator = test_datagen.flow_from_directory(
    config.base_path,
    color_mode='rgb',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=0
    )

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    print(labels)
    
    #################################
    # Save Class IDs
    #################################
    classes_json = train_generator.class_indices
    num_classes = len(train_generator.class_indices)

    with open(config.model.save_dir + 'classes.json', 'w') as fp:
        json.dump(classes_json, fp, indent = 4)
    
    
    #setting mlflow tracking information. This should be modified depending on mlflow configurations
    #mlflow.set_tracking_uri('mysql+pymysql://root:root123@localhost:33060/mlflow')
    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
    
    print( 'file://' + utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(config.mlflow.experiment_name)
    print(tf.__version__)
    with mlflow.start_run():
        
        selected_optimizer = get_optimizer(config.model.hyperparameters.optimizer_method, config.model.hyperparameters.learning_rate)

        # Pre-defined model pools
        models = ['CNN1','CNN2']
        for name in models:
            i=0
            if (i==0):
             model = test_model(selected_optimizer, input_shape)
            else:
                model = test_model2(selected_optimizer, input_shape)
                
            i+=1 
            model.summary()
            model.compile(
                         loss = config.model.hyperparameters.loss,
                         optimizer = selected_optimizer,
                         metrics = config.model.hyperparameters.metrics
                        )
#
            print(config.model.hyperparameters)
            param_grid = dict(config.model)

            for key, value in param_grid.items():
                   # MLFlow tracking - use log_param() to track hyper-parameters 
                   mlflow.log_param(key, value)
 
            history= model.fit_generator(train_generator, epochs=config.model.hyperparameters.epochs, validation_data=validation_generator)

            for key, values in history.history.items():
                for i, v in enumerate(values):
                
        # use log_metric() to track evaluation metrics
                    mlflow.log_metric(f'{name}_'+ key, v, step=i)
                # use log_metric() to track evaluation metrics
            

        #for i, layer in enumerate(model.layers):
            # use log_param() to track model.layer (details of each CNN layer)
            #mlflow.log_param(f'hidden_layer_{i}_units', layer.output_shape)
      
      
       # use log_model() to track output Keras model (accessible through the MLFlow UI)
            log_model(model, f'lego_cnnmodel_{name}')
  
    
       ## sketch loss
            fig = show_final_history(history)
            mlflow.log_artifact(config.model.save_dir + 'classes.json')
    # save loss picture, use log_artifact() to track it in MLFLow UI
            fig.savefig(f'{name}_train-validation-loss.png')
            mlflow.log_artifact(f'{name}_train-validation-loss.png')
        

if __name__== "__main__":
    main()