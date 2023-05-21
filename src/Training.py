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
    test_seq = []
    test_label = []
    val_seq = [] 
    val_label = []
    
    train_seq = np.load('../data/split-train-val-test/train-features-scaled.npz')['arr_0']
    train_label = np.load('../data/split-train-val-test/train-labels.npz')['arr_0']
    
    val_seq = np.load('../data/split-train-val-test/validation-features-scaled.npz')['arr_0']
    val_label = np.load('../data/split-train-val-test/validation-labels.npz')['arr_0']
    
    test_seq = np.load('../data/split-train-val-test/test-features-scaled.npz')['arr_0']
    test_label = np.load('../data/split-train-val-test/test-labels.npz')['arr_0']

    # Create and train the model
    model = create_model(train_seq[0].shape)
    model.summary()
    
    history = History()
    model_checkpoint = ModelCheckpoint(filepath = save_file_path + 'checkpoints/tada.{epoch:02d}-{val_loss:.2f}.hdf5',\
            save_weights_only=True,\
                monitor='val_loss',\
                    mode='auto',\
                        save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_f1_metric', patience=7, verbose=1)
    
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
    model.save('classification_tada.h5')

    
    # Plot metrics
    model_history = pd.DataFrame(history.history)
    training_precision = model_history['precision']#.values()
    validation_precision = model_history['val_precision']#.values()
    plot_metric(training_precision, validation_precision, 'precision')
    
    training_f1 = model_history['f1_metric']#.values()
    validation_f1 = model_history['val_f1_metric']#.values()
    plot_metric(training_f1, validation_f1, 'f1_metric')
    
    training_accuracy = model_history['accuracy']#.values()
    validation_accuracy = model_history['val_accuracy']#.values()
    plot_metric(training_accuracy, validation_accuracy, 'accuracy')

