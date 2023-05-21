# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:44:10 2023

@author: lisav
"""

import csv
from pickle import load
from sklearn.decomposition import KernelPCA
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import openTSNE
from Preprocessing import scale_features
from Preprocessing import class1_extractions

plt.rcParams['figure.dpi'] = 600
save_path_file = "../data/clustering/"
if not os.path.exists(save_path_file):
     os.mkdir(save_path_file)
        
'''
Load scaled features and select class 1

'''
print("Load features, AD class, and shap values")
features = load(open('../data/features_OnlyPlants.pkl', 'rb'))

with open('../data/TrainingsData.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = []
    for i in csv_reader:
        data.append([i[0], i[1], i[2], i[3], i[4]])
data.pop(0)
ad_class = np.double(np.array(data)[:, 4])

class1_features, class1_idx = class1_extractions(ad_class, features)

shap_values = load(open('../data/SHAP/shap_values_class1_ADs_plantonlydata.pkl', 'rb'))
class1_shap_values = shap_values[0]

'''
Select top 8  important features

'''
NUM_FEAT = 8

with open('../data/SHAP/feature-importance-average.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    feature_importance = []
    for i in csv_reader:
        feature_importance.append([i[0], float(i[1])])
feature_importance.pop(0)

feature_importance.sort(key = lambda x: x[1], reverse = True)
idx = np.array(feature_importance)[:NUM_FEAT,0]

class1_features_sub = class1_features[:,:,list(map(int, list(idx)))]
class1_shap_values_sub = class1_shap_values[:,:,list(map(int, list(idx)))]

class1_features_sub = scale_features(class1_features_sub)

'''
Perform PCA

'''

def clustering(features, shap_values, n_components = 10, FEATURES = 8, n_clusters = 6): 
    
    '''
    input: features as 3D array and SHAP values
    reduces the subsequence dimensions to 1 to output a 2D array
    concatenates features with SHAP values
    reduces the dimension to 10
    reduces the dimensions to 2 with t-SNE
    
    output: 2D array for plotting
    
    '''
    
    features_flat = np.zeros((features.shape[0], 1, FEATURES))
    for i in range(FEATURES):
        kpca = KernelPCA(n_components=1, random_state=2) 
        kpca2 = kpca.fit_transform(features[:, :, i]) 
        print(kpca.eigenvalues_)
        features_flat[:, :, i] = kpca2
    
    features_flat = np.reshape(features_flat, [features.shape[0], FEATURES])
    
    shap_values_flat = []
    for sequence in shap_values:
        shap_values_flat.append(np.sum(sequence, axis = 0))
    shap_values_flat = pd.DataFrame(shap_values_flat)
    
    concat = np.concatenate((shap_values_flat, features_flat), axis = 1)
    
    pca = KernelPCA(n_components, random_state=2) 
    concat_10 = pca.fit_transform(concat)
    
    pcaInit = concat_10[:,:2] / np.std(concat_10[:,0]) * 0.0001

    tsne = openTSNE.TSNE(random_state = 42, initialization = pcaInit,
                         learning_rate = features.shape[0]/12,
                         perplexity = [30, int(features.shape[0]/100)],
                         metric="cosine",
                         early_exaggeration = features.shape[0]/100)
    tsne_result = tsne.fit(concat_10)
    
    kmeans = KMeans(n_clusters, init="random", n_init=10, random_state=1)
    label = kmeans.fit_predict(tsne_result)
    
    return tsne_result, label

def figure_tsne(tsne_result, label = None): 
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1]})
    
    #pal = sns.color_palette('tab20')
    pal = ['#e377c2', '#ff7f0e', '#bcbd22', '#2ca02c', '#17becf', '#9467bd']
    fig, ax = plt.subplots(1)
    if label is None:
        sns.scatterplot(x='tsne_1', y='tsne_2', data=tsne_result_df, ax=ax,s=1.5)
    else:
        tsne_result_df = tsne_result_df.join(pd.DataFrame(data=label, columns = ['label']))
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette=pal, data=tsne_result_df, ax=ax,s=4) 
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    return fig

def elowplot(tsne_result):
    '''

    Parameters
    ----------
    tsne_result : a 2D array

    Returns
    -------
    None.

    '''
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init="random", n_init=10, random_state=1)
        kmeans.fit(tsne_result)
        sse.append(kmeans.inertia_)

    #visualize results
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    
    return plt

tsne_result, label = clustering(class1_features_sub, class1_shap_values_sub, FEATURES = 8, n_clusters = 6)
figure = figure_tsne(tsne_result, label)
elowplot(tsne_result)

data_class1 = np.array(data)[class1_idx]
data_class1 = data_class1[:, [1,2,4]]

data_tsne = pd.DataFrame(tsne_result, columns = ['tsne-1', 'tsne-2']).join(pd.DataFrame(label, columns = ['label']))
data_tsne = data_tsne.join(pd.DataFrame(class1_idx, columns = ['idx']))
data_tsne = data_tsne.join(pd.DataFrame(data_class1, columns = ['name', 'scale_score', 'sequence']))

data_tsne.to_csv(save_path_file + "tsne-clustering.csv")

figure.savefig(save_path_file + 't-sne_6clusters.pdf',dpi=600)
