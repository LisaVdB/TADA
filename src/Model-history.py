# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:41:16 2021

@author: lisav
"""

import pickle
import csv

load_file_path = 'D:/Google_Drive/Post-doc/Project_ARFs/ARFs/Classification_TRIPP/'
global model_history

with open(load_file_path + 'data/model-results/model-history_28epochs.json', 'rb') as f:
    model_history = pickle.load(f)

with open(load_file_path + 'data/model-results/output_history_28epochs.csv', 'w') as output:
    writer = csv.writer(output)
    for key, value in model_history.items():
        writer.writerow([key, value])
    output.close()    

with open(load_file_path + 'data/model-results_OnlyPlants/model-history_28epochs.json', 'rb') as f:
    model_history_OnlyPlants = pickle.load(f)

with open(load_file_path + 'data/model-results_OnlyPlants/output_history_OnlyPlants_28epochs.csv', 'w') as output:
    writer = csv.writer(output)
    for key, value in model_history_OnlyPlants.items():
        writer.writerow([key, value])
    output.close()    

with open(load_file_path + 'data/model-results_OnlyRP/model-history.json', 'rb') as f:
    model_history_OnlyRandom = pickle.load(f)

with open(load_file_path + 'data/model-results_OnlyRP/output_history_OnlyRandom.csv', 'w') as output:
    writer = csv.writer(output)
    for key, value in model_history_OnlyRandom.items():
        writer.writerow([key, value])
    output.close()  

#test results sequences
with open(load_file_path + 'data/model-results/test-set-results.json', 'rb') as f:
    test_results = pickle.load(f)
with open(load_file_path + 'data/model-results_OnlyPlants/test-set-results.json', 'rb') as f:
    test_results_OnlyPlants = pickle.load(f)
with open(load_file_path + 'data/model-results_OnlyRP/test-set-results.json', 'rb') as f:
    test_results_OnlyRP = pickle.load(f)
with open(load_file_path + 'data/model-results_OnlyPlantsOversample/test-set-results.json', 'rb') as f:
    test_results_OnlyPlantsOversample = pickle.load(f)