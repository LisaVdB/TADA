# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:32:23 2023

@author: lisav
"""

import csv
import numpy as np
from pickle import dump, load
import os
from sklearn.model_selection import train_test_split

from Preprocessing import scale_features
from Preprocessing import create_features

np.random.seed(1258)  # for reproducibility
save_file_path = '../data/split-train-val/'
if not os.path.exists(save_file_path):
     os.mkdir(save_file_path)

'''
Data contains four distinct datasets 

'''
print('Loading data')
with open('../data/TrainingsData.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = []
    for i in csv_reader:
        data.append([i[0], i[1], i[2], i[3], i[4]])
data.pop(0)

sequences = [i[2] for i in data]
data = np.array(data)
y0 = np.double(data[:, 4])
y = np.column_stack([y0, 1 - y0])

'''
Splitting into train, and validation set 

'''
# Save the features and activation scores
features = load(open('../data/features_OnlyPlants.pkl', 'rb'))

# Split sequences in training and testing data
X_train, X_val, y_train, y_val = train_test_split(features, y, random_state = 42, test_size=0.1, stratify = y)

# Scale features
X_train_scaled = scale_features(X_train)
X_val_scaled = scale_features(X_val)
print(X_train_scaled[0].shape)

print('Saving train-validation-test sequences and labels')
np.savez_compressed(save_file_path + 'train-features-scaled.npz', X_train_scaled)
np.savez_compressed(save_file_path + 'train-labels.npz', y_train)

np.savez_compressed(save_file_path + 'validation-features-scaled.npz', X_val_scaled)
np.savez_compressed(save_file_path + 'validation-labels.npz', y_val)
