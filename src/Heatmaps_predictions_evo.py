# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:33:30 2023

@author: lisav
"""


import os
os.chdir("D:/Google_Drive/Post-doc\\Project_ARFs\\ARFs\\Classification_TRIPP\\src\\")

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams['figure.dpi'] = 600
save_path_file = "../../Predictions_TRIPP/EvolutionLib11/"

with open(save_path_file + 'AARFCladeDataforLisa.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    df = []
    for i in csv_reader:
        df.append([i[0], i[1], i[2], i[3], i[4], i[5], i[6]])
df = pd.DataFrame(df[1:], columns=df[0])
df['Predictions'] = df['Predictions'].astype(float)
df['norm'] = df['norm'].astype(float)

nl=['ARF5', 'ARF7', 'ARF6', 'ARF8']
ml=['ARF','AL','Brara','Med','KHR','Sl','Mig','Sobic','Zm','LOC','Bradi','evm']
for i in nl:
    data=df.loc[df.Clade==i]
    test={}
    gl=[]
    for n in ml:
        gl.extend(data.loc[data.Name.str.startswith(n),'Model'].unique().tolist())
    for j in gl:
        temp=[]
        for z in np.linspace(0,.95,20):
            temp.append(np.mean(data.loc[(data.Model==j)&(data.norm>=z)&(data.norm<(z+.05)),'Predictions']))
        test[j]=temp
    data=pd.DataFrame(test)
    data=data.transpose()
    palette = sns.light_palette(color = "#E377C2", as_cmap=True)
    palette = sns.blend_palette(colors=('#FFFFFF','#e377c2'),as_cmap=True)
    plt.figure()
    sns.heatmap(data,vmin=0.3,vmax=0.7,cmap=palette)
    plt.title(i)
    plt.savefig(save_path_file + 'heatmap_'+ i +'_Clade.pdf',dpi=300,bbox_inches="tight",pad_inches=0.1)