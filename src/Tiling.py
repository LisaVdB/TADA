# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:24:45 2023

@author: Nickm
"""
import pandas as pd
from Bio import SeqIO

save_file_path = '../data/predictions/'

filename = "Fasta File Goes Here"
WindowSize = 40
WindowSpacing = 10

proteins_full = []
proteins_chopped = []
proteins_chopped_names = []

for record in SeqIO.parse(filename, "fasta"):
    tempProtein = str(record.seq)   
    proteins_full.append(tempProtein)
    Nwindows = len(tempProtein) - WindowSize + WindowSpacing
    i = 0
    while i < len(tempProtein) - WindowSize:
        proteins_chopped.append(tempProtein[i:i+WindowSize])
        proteins_chopped_names.append(record.id + '_%i'%(i+1))
        i += WindowSpacing
    lastseq=tempProtein[-WindowSize:]
    proteins_chopped.append(lastseq)
    proteins_chopped_names.append(record.id + '_%i'%(len(tempProtein)-WindowSize+1))
    
tempDF = pd.DataFrame({'ADseq':proteins_chopped,'Name':proteins_chopped_names})
tempDF.to_csv(save_file_path + filename + "_tiles.csv")
