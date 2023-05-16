# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:04:14 2023

@author: lisav
"""
import os
os.chdir("D:/Google_Drive/Post-doc/Project_ARFs/ARFs/Classification_TRIPP/src")

from Model import create_model
from Preprocessing import class1_extractions
from Preprocessing import scale_features
import csv
import numpy as np
from pickle import dump, load
import shap
import pandas as pd
np.random.seed(1258)  # for reproducibility


'''
Import data
'''
save_file_path = '../data/SHAP/'

with open('../data/TrainingsData_OnlyPlants.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = []
    for i in csv_reader:
        data.append([i[0], i[2], i[4], i[5], i[6]])
data.pop(0)
ad_class = np.double(np.array(data)[:, 3])

features = load(open('../data/features_OnlyPlants.pkl', 'rb'))


'''
Extract true positive predictions
'''

class1_features, class1_idx = class1_extractions(ad_class, features)
class1_features = scale_features(class1_features)
    
'''
Load model
'''

model = create_model(SHAPE = (36, 42))
print('\x1b[2K\tModel created')

model_weights_path = '../data/model-results-notest/checkpoints/'
model.load_weights(model_weights_path + 'tripp.14-0.02.hdf5')
print('\x1b[2K\tWeights loaded')

predictions = model.predict(class1_features)

dump(predictions, open('predictions_class1_ADs_plantonlydata.pkl', 'wb'))
predictions = load(open('predictions_class1_ADs_plantonlydata.pkl', 'rb'))

'''
Run SHAP
'''
X_test = class1_features[:10]

e = shap.GradientExplainer(model, X_test)
shap_values = e.shap_values(X_test)
dump(shap_values, open(save_file_path + 'shap_values_class1_ADs_plantonlydata.pkl', 'wb'))

print(len(shap_values), len(shap_values[0]), len(shap_values[0][0]),len(shap_values[0][0][0]))

FEATURES = 42
#Calculate the sum of each feature across the 36 winows and all the class1 sequences
feature_importance = [0 for i in range(FEATURES)] #empty list len=42
for sequence in shap_values[0]: 
    for window in sequence: 
        for i in range(len(window)):
            feature_importance[i]+=abs(window[i])

data = pd.DataFrame(feature_importance)
#data.columns = ["index", "predictions"]
data.to_csv(save_file_path + "feature-importance-average.csv")

#Calculate the sum of each feature across the 36 winows -- keep sequences for variance
feature_importance = []
for sequence in shap_values[0]: 
    feature_importance.append(np.sum(abs(sequence), axis = 0))

data = pd.DataFrame(feature_importance)
#data.columns = ["index", "predictions"]
data.to_csv(save_file_path + "feature-importance.csv")