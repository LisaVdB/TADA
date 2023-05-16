from Preprocessing import scale_features
from Preprocessing import create_features
from Preprocessing import split_seq
from Model import create_model
import csv
import pandas as pd
import numpy as np
from pickle import dump, load
np.random.seed(1258)  # for reproducibility


'''
Open the list of sequences and their activation scores
This code assumes a csv with three columns: labels, sequences, score, and class
'''

save_file_path = '../data/predictions/'
with open(save_file_path + 'TFValidationListForLisa.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    data = []
    for i in csv_reader:
        data.append([i[0], i[1], i[2], i[3]])
data.pop(0)

labels = [i[0] for i in data]
sequences = [i[1] for i in data]
y0 = np.double(np.array(data)[:, 3])
y = np.column_stack([y0, 1 - y0])

'''
Calculate features
'''

# Defines the sequence window size and steps (stride length). Change values if needed.
SEQUENCE_WINDOW = 5
STEPS = 1
LENGTH = 40

#Calculate and scale features for larger sequences
#sequences_40aa_center = split_seq(sequences, labels, "center")
#sequences_40aa_start = split_seq(sequences, labels, "start")
#sequences_40aa_end = split_seq(sequences, labels, "end")
#sequences_40aa = [a+b+c for a,b,c in zip(sequences_40aa_center, sequences_40aa_start, sequences_40aa_end)]
#labels_40aa = sequences_40aa[1]
#sequences_40aa = sequences_40aa[0]
#y_40aa = list(y) * 3 #For oversampling

features = create_features(sequences, SEQUENCE_WINDOW, STEPS)
features_scaled = scale_features(features)

# Save the features
dump(features_scaled, open(save_file_path + 'features_Library11_evolution.pkl', 'wb')) #Change if needed

#When features are already generated
features_scaled = load(open(save_file_path + 'features_TFvalidation.pkl', 'rb'))
#y = load(open('scores_yeastTFs.pkl', 'rb'))

'''
Load model
'''

model = create_model(SHAPE = (36, 42))
print('\x1b[2K\tModel created')

model_weights_path = '../data/model-results-notest/checkpoints/'
model.load_weights(model_weights_path + 'tripp.14-0.02.hdf5')
print('\x1b[2K\tWeights loaded')

#Make classification predictions
results = model.evaluate(features_scaled, y, verbose = 1)
print(model.metrics_names)
predictions = model.predict(features_scaled)

'''
Load model for plants only data
'''
#model_OnlyPlants = create_model(SHAPE = (36, 42))
#print('\x1b[2K\tModel created')

#model_weights_path = '../data/model-results_OnlyPlants/checkpoints/'
#model_OnlyPlants.load_weights(model_weights_path + 'tripp.19-0.02.hdf5')
#print('\x1b[2K\tWeights loaded')
#Make classification predictions (only plants)
#model_OnlyPlants = load_model('../data/model-results_OnlyPlants/classification_tripp.h5', custom_objects={'f1_metric': f1_metric})
#predictions_OnlyPlants = model_OnlyPlants.predict(features_scaled)

'''
Save data
'''

#data = list(zip(labels_40aa, sequences_40aa, list(y_40aa), list(predictions[:,0])))
#data = pd.DataFrame(data)
#data.columns = ["labels", "sequences", "scaled_score", "predictions"]
#data.to_csv(save_file_path + "Sanborn_40aa.csv")

data = list(zip(labels, sequences, list(y), list(predictions[:,0])))
data = pd.DataFrame(data)
data.columns = ["labels", "sequences", "scaled_score", "predictions"]
data.to_csv(save_file_path + "TFValidationPredictions.csv")
