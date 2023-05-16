# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:49:01 2023

@author: lisav
"""

import os
os.chdir("D:/Google_Drive/Post-doc/Project_ARFs/ARFs/Classification_TRIPP/src")

import csv
from pickle import load
import numpy as np
import pandas as pd
from Model import create_model
from Preprocessing import scale_features
import shap

save_file_path = "../data/clustering/"

'''
Create array of the features for each of the five clusters

'''
with open('../data/clustering/tsne-clustering_5clusters.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = []
    for i in csv_reader:
        data.append([i[3], i[4], i[5], i[6]])
data.pop(0)

cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []
cluster5 = []

for i in range(len(data)):
    if data[i][0] == "0":
        cluster1.append(data[i])
    if data[i][0] == "1":
        cluster2.append(data[i])
    if data[i][0] == "2":
        cluster3.append(data[i])
    if data[i][0] == "3":
        cluster4.append(data[i])
    if data[i][0] == "4":
        cluster5.append(data[i])
        
features = load(open('../data/features_OnlyPlants.pkl', 'rb'))

features_c1 = scale_features(features[list(map(int, list(np.array(cluster1)[:,1])))])
features_c2 = scale_features(features[list(map(int, list(np.array(cluster2)[:,1])))])
features_c3 = scale_features(features[list(map(int, list(np.array(cluster3)[:,1])))])
features_c4 = scale_features(features[list(map(int, list(np.array(cluster4)[:,1])))])
features_c5 = scale_features(features[list(map(int, list(np.array(cluster5)[:,1])))])

'''
Run SHAP on each of the five clusters

'''

model = create_model(SHAPE = (36, 42))
print('\x1b[2K\tModel created')

model_weights_path = '../data/model-results-notest/checkpoints/'
model.load_weights(model_weights_path + 'tripp.14-0.02.hdf5')
print('\x1b[2K\tWeights loaded')

e = shap.GradientExplainer(model, features_c1)
shap_values_c1 = e.shap_values(features_c1)
e = shap.GradientExplainer(model, features_c2)
shap_values_c2 = e.shap_values(features_c2)
e = shap.GradientExplainer(model, features_c3)
shap_values_c3 = e.shap_values(features_c3)
e = shap.GradientExplainer(model, features_c4)
shap_values_c4 = e.shap_values(features_c4)
e = shap.GradientExplainer(model, features_c5)
shap_values_c5 = e.shap_values(features_c5)


'''
Compute and visualize average feature across subsequences for the five clusters

'''

def shap_average(shap_values):
    FEATURES = 42
    feature_importance_avg = [0 for i in range(FEATURES)]
    for sequence in shap_values[0]: 
        for window in sequence: 
            for i in range(len(window)):
                feature_importance_avg[i]+=abs(window[i])
    
    feature_importance = []
    for sequence in shap_values[0]: 
        feature_importance.append(np.sum(abs(sequence), axis = 0))
        
    return feature_importance_avg, feature_importance

feature_importance_avg_c1, feature_importance_c1 = shap_average(shap_values_c1)
feature_importance_avg_c2, feature_importance_c2 = shap_average(shap_values_c2)
feature_importance_avg_c3, feature_importance_c3 = shap_average(shap_values_c3)
feature_importance_avg_c4, feature_importance_c4 = shap_average(shap_values_c4)
feature_importance_avg_c5, feature_importance_c5 = shap_average(shap_values_c5)

data = pd.DataFrame(feature_importance_c1)
data.to_csv(save_file_path + "feature-importance-c1.csv")
data = pd.DataFrame(feature_importance_c2)
data.to_csv(save_file_path + "feature-importance-c2.csv")
data = pd.DataFrame(feature_importance_c3)
data.to_csv(save_file_path + "feature-importance-c3.csv")
data = pd.DataFrame(feature_importance_c4)
data.to_csv(save_file_path + "feature-importance-c4.csv")
data = pd.DataFrame(feature_importance_c5)
data.to_csv(save_file_path + "feature-importance-c5.csv")

data = pd.DataFrame(feature_importance_avg_c1)
data.to_csv(save_file_path + "feature-importance-avg-c1.csv")
data = pd.DataFrame(feature_importance_avg_c2)
data.to_csv(save_file_path + "feature-importance-avg-c2.csv")
data = pd.DataFrame(feature_importance_avg_c3)
data.to_csv(save_file_path + "feature-importance-avg-c3.csv")
data = pd.DataFrame(feature_importance_avg_c4)
data.to_csv(save_file_path + "feature-importance-avg-c4.csv")
data = pd.DataFrame(feature_importance_avg_c5)
data.to_csv(save_file_path + "feature-importance-avg-c5.csv")
