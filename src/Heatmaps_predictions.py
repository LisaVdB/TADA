# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:55:54 2023

@author: Nick
"""

import os
os.chdir("D:/Google_Drive/Post-doc\\Project_ARFs\\ARFs\\Classification_TRIPP\\src\\")

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams['figure.dpi'] = 600
save_path_file = "../../Predictions_TRIPP/ValidationTFs/"

with open(save_path_file + 'TFValidationPredictions.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    df = []
    for i in csv_reader:
        df.append([i[1], i[2], i[3], i[4]])
df = pd.DataFrame(df[1:], columns=df[0])
df[['ATGNum', 'start']] = df['labels'].str.split('_', expand=True)
df['start_int'] = df['start'].astype(int)
df['predictions'] = df['predictions'].astype(float)

l = df['ATGNum'].drop_duplicates()

data_all = []

#l=['AT1G09530.1']
for k in l:
    data=df.loc[df.ATGNum==k]
    plt.figure()
    ss={}
    for i in data.start_int.values:
        for j in np.linspace(i,i+39,40):
            if j in ss: 
                ss[j].append(data.loc[data.start_int==i,'predictions'].values[0])
            else:
                ss[j]=[data.loc[data.start_int==i,'predictions'].values[0]]
    temp=[]
    for i in ss:
        y=np.mean(ss[i])
        temp.append(y)
    data=pd.DataFrame(temp)
    data=data.transpose()
    
    data_all.append(data)
    
    palette = sns.light_palette(color = "#E377C2", n_colors = 20, as_cmap=True) 
    #["#FFFFFF", "#E377C2", "#871C67"]
    sns.heatmap(data,cmap = palette, vmax=0.7,vmin=0.3)
    plt.title(k)
    plt.savefig(save_path_file + k +'_heatmap.pdf',dpi=300,bbox_inches="tight",pad_inches=0.1)
    
