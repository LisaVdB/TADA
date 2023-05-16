# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:48:10 2022

@author: lisav
"""

from Model import plot_metric
from Model import create_model

import csv
import numpy as np
import pandas as pd
from pickle import dump, load
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from sklearn.utils import class_weight

np.random.seed(1258)  # for reproducibility

def main():
    '''
    loading saved-data batch
    '''
    
    # change last folder name by respective batches
    save_file_path = '../data/model-results/'
    
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)
        os.mkdir(save_file_path + 'checkpoints/')
        print('-----Directory made-----')
    else:
        print('-----Directory already exists-----')
        
    print('Loading train-validation-test sequences and labels!')

    # load 1st batch data
    train_seq = []
    train_label = []
    val_seq = [] 
    val_label = []
    
    train_seq = np.load('../data/split-train-val-test/train-features-scaled.npz')['arr_0']
    train_label = np.load('../data/split-train-val-test/train-labels.npz')['arr_0']
    
    val_seq = np.load('../data/split-train-val-test/validation-features-scaled.npz')['arr_0']
    val_label = np.load('../data/split-train-val-test/validation-labels.npz')['arr_0']

    # Create and train the model
    model = create_model(train_seq[0].shape)
    model.summary()
    
    history = History()
    model_checkpoint = ModelCheckpoint(filepath = save_file_path + 'checkpoints/tripp.{epoch:02d}-{val_loss:.2f}.hdf5',\
            save_weights_only=True,\
                monitor='val_loss',\
                    mode='auto',\
                        save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    callbacks = [
        history,
        model_checkpoint,
        earlystopper
    ]

    #class weights
    ground_truth = np.argmax(train_label, axis=-1)
    class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(ground_truth), \
        y = ground_truth)
    d_class_weights = dict(enumerate(class_weights))
    
    history = model.fit(train_seq, train_label, batch_size=64,
                        epochs=20, verbose=1, callbacks=callbacks,
                        class_weight = d_class_weights,
                        validation_data=(val_seq, val_label))
    model.save('classification_tripp.h5')


