# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:59:00 2022

"""

import numpy as np
from copy import deepcopy
from pickle import dump
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from localcider.sequenceParameters import SequenceParameters
import alphaPredict as alpha

# Returns the middle 30AA sequences from longer sequences
def split_seq(sequences, labels, position, length_out = 40, length_min = 20):
    '''

    Parameters
    ----------
    sequences : List.
        List of sequences.
        Variable length of the input sequences is allowed.
        Smaller than 40AA length will be padded with M
    labels : list
        List of labels corresponding to the sequences
    Position : String
        "center", "start", "end". Determines the position of the 40AA seq
    Returns
    -------
    sequences_40aa : List.
        Option center: 40 AA sequences centered around the middle of the input sequences.
        Option start: first 40 AA of the input sequences.
        Option end: last 40 AA of the input sequences.

    '''
    sequences_40aa = []
    labels_40aa = []
    for i, seq in enumerate(sequences):
        if len(seq) < length_min:
            sequences_40aa = sequences_40aa
            labels_40aa = labels_40aa
        elif (len(seq) >= length_min) & (len(seq) < length_out):
            sequences_40aa.append({'end': seq.ljust, 'center': seq.center, 'start': seq.rjust}[position](40, "M"))
            labels_40aa.append(labels[i])
        elif len(seq) == length_out:
            sequences_40aa.append(seq)
            labels_40aa.append(labels[i])
        elif len(seq) > length_out: 
            SIDE = round((len(seq)-length_out)/2)
            labels_40aa.append(labels[i])
            if position == "center":
                sequences_40aa.append(seq[SIDE:SIDE+length_out])
            elif position == "start":
                sequences_40aa.append(seq[:length_out])
            elif position == "end":
                sequences_40aa.append(seq[:-length_out])
    return sequences_40aa, labels_40aa


# Retrieves the various properties of the sequences with localCIDER and assigns them to the features array
def create_features(sequences, SEQUENCE_WINDOW = 5, STEPS = 1, LENGTH = 40, PROPERTIES = 42):
    '''
    Parameters
    ----------
    sequences : List. 
        List of sequences with max length of 30AA.
    SEQUENCE_WINDOW : Int, optional
        DESCRIPTION. The default is 5.
    STEPS : Int, optional
        DESCRIPTION. The default is 1.
    PROPERTIES : Int, optional
        DESCRIPTION. The default is 21.

    Returns
    -------
    features : Processed input data for model

    '''

    aliphatics = ['I','V','L','A']
    aromatics = ['W','F','Y']
    branching = ['V','I','T']
    charged = ['K','R','H','D','E']
    negatives = ['D','E']
    phosphorylatables = ['S','T','Y']
    polars = ['R','K','D', 'E', 'Q', 'N', 'Y']
    hydrophobics = ['W','F','L','V', 'I', 'C', 'M']
    positives = ['K','R','H']
    sulfurcontaining = ['M','C']
    tinys = ['G','A','S','P']
    amino_acids = ['R', 'K', 'D', 'E', 'Q', 'N', 'H', 'S', 'T', 'Y', 'C', 'W', 'M', 'A', 'I', 'L', 'F', 'V', 'P', 'G']
    features = []        
    for i, sequence in enumerate(sequences):
        SEQUENCE_LENGTH = len(sequence)
        SeqOb = SequenceParameters(sequence)
        kappa = np.full(int((SEQUENCE_LENGTH-SEQUENCE_WINDOW)/STEPS+1), SeqOb.get_kappa())
        omega = np.full(int((SEQUENCE_LENGTH-SEQUENCE_WINDOW)/STEPS+1), SeqOb.get_Omega())
        
        sub_seq = np.array([SequenceParameters(sequence[STEPS*j:(STEPS*j+SEQUENCE_WINDOW)]) for j in range(int((SEQUENCE_LENGTH-SEQUENCE_WINDOW)/STEPS+1))])
        hydropathy = np.array(list(map(lambda x: x.get_mean_hydropathy(),sub_seq))) 
        hydropathy_ww = np.array(list(map(lambda x: x.get_WW_hydropathy(),sub_seq))) 
        ncpr = np.array(list(map(lambda x: x.get_NCPR(),sub_seq)))
        promoting = np.array(list(map(lambda x: x.get_fraction_disorder_promoting(),sub_seq)))
        fcr = np.array(list(map(lambda x: x.get_FCR(),sub_seq)))
        charge = np.array(list(map(lambda x: x.get_mean_net_charge(),sub_seq)))
        negative = np.array(list(map(lambda x: x.get_fraction_negative(),sub_seq)))
        positive = np.array(list(map(lambda x: x.get_fraction_positive(),sub_seq)))
        
        sub_seq = np.array([sequence[STEPS*j:(STEPS*j+SEQUENCE_WINDOW)] for j in range(int((SEQUENCE_LENGTH-SEQUENCE_WINDOW)/STEPS+1))])
        one = np.array(list(map(lambda x: len(list(filter(aliphatics.__contains__, x))), sub_seq)))
        two = np.array(list(map(lambda x: len(list(filter(aromatics.__contains__, x))), sub_seq)))
        three = np.array(list(map(lambda x: len(list(filter(branching.__contains__, x))), sub_seq)))
        four = np.array(list(map(lambda x: len(list(filter(charged.__contains__, x))), sub_seq)))
        five = np.array(list(map(lambda x: len(list(filter(negatives.__contains__, x))), sub_seq)))
        six = np.array(list(map(lambda x: len(list(filter(phosphorylatables.__contains__, x))), sub_seq)))
        seven = np.array(list(map(lambda x: len(list(filter(polars.__contains__, x))), sub_seq)))
        eight = np.array(list(map(lambda x: len(list(filter(hydrophobics.__contains__, x))), sub_seq)))
        nine = np.array(list(map(lambda x: len(list(filter(positives.__contains__, x))), sub_seq)))
        ten = np.array(list(map(lambda x: len(list(filter(sulfurcontaining.__contains__, x))), sub_seq)))
        eleven = np.array(list(map(lambda x: len(list(filter(tinys.__contains__, x))), sub_seq)))
        sstructure = np.array(list(map(lambda x: sum(alpha.predict(x))/len(x), sub_seq)))
        count_20 = [[s.count(aa) for s in sub_seq] for aa in amino_acids]

        x = np.array([kappa, omega, hydropathy, hydropathy_ww, ncpr, promoting, fcr, charge, negative, positive,
                     one, two, three, four, five, six, seven, eight, nine, ten, eleven, sstructure])
        x = np.concatenate([x, count_20])
        shape = np.shape(x)
        padded_x = np.zeros((PROPERTIES, int((LENGTH-SEQUENCE_WINDOW)/STEPS+1)))
        padded_x[:shape[0],:shape[1]] = x
        features.append(padded_x)
        if i%1000==0: print(i)
    features = np.array(features)
    features = np.transpose(features, (0, 2, 1))
    return features

# Scale scores
def scale_y(y):
    '''
    Parameters
    ----------
    y : Array of float.

    Returns
    -------
    y_scaled : Array of float.

    '''
    scaler = MinMaxScaler((-1,1))
    dump(scaler, open('scaler_y.pkl', 'wb'))
    y_scaled = scaler.fit_transform(y.reshape(-1,1))
    return y_scaled
# Scale features
def scale_features(features: np.ndarray, SEQUENCE_WINDOW = 5, STEPS = 1, LENGTH = 40) -> np.ndarray:
    '''
    Parameters
    ----------
    features : np.ndarray
        Takes the output of create_features() and scales the values per feature column.

    Returns
    -------
    scaled_array_copy : TYPE
        DESCRIPTION.

    '''
    scaler = StandardScaler()
    scaler2 = MinMaxScaler()
    
    scaled_array_copy = deepcopy(features)
    n, m = features[0].shape
    scaler_metric = np.array(np.zeros((len(features[0][0]), 7)))

    for i in range(m):
        results = scaler.fit_transform(scaled_array_copy[:,:,i].reshape(-1,1)).reshape(-1)
        scaler_metric[i, :3] = np.array([scaler.mean_, scaler.var_, scaler.scale_]).reshape(-1)
        results = scaler2.fit_transform(results.reshape(-1,1)).reshape(-1)
        scaler_metric[i, 3:7] = np.array([scaler2.min_, scaler2.data_min_, scaler2.data_max_, scaler2.scale_]).reshape(-1)
        results = results.reshape((len(features),int((LENGTH-SEQUENCE_WINDOW)/STEPS+1)))
        for j, result in enumerate(results):
            scaled_array_copy[j,:,i] = result
            
    dump(scaler_metric, open('scaler_metric.arr', 'wb'))
    
    return scaled_array_copy

# Scale features for prediction
def scale_features_predict(features: np.ndarray, SEQUENCE_WINDOW = 5, STEPS = 1, LENGTH = 40) -> np.ndarray:
    '''
    Parameters
    ----------
    features : np.ndarray
        Takes the output of create_features() and scales the values per feature column.

    Returns
    -------
    scaled_array_copy : TYPE
        DESCRIPTION.

    '''
    scaler = StandardScaler()
    scaler2 = MinMaxScaler()
    
    scaled_array_copy = deepcopy(features)
    n, m = features[0].shape
    scaler_metric = np.load('scaler_metric.arr', allow_pickle=True)

    for i in range(m):
        scaler.mean_, scaler.var_, scaler.scale_, scaler.n_samples_seen_ = scaler_metric[i,:4]
        results = scaler.transform(scaled_array_copy[:,:,i].reshape(-1,1)).reshape(-1)
        scaler2.min_, scaler2.data_min_, scaler2.data_max_, scaler2.scale_, scaler2.n_samples_seen_, scaler2.data_range_ = scaler_metric[i, 4:10]
        results = scaler2.transform(results.reshape(-1,1)).reshape(-1)
        results = results.reshape((len(features),int((LENGTH-SEQUENCE_WINDOW)/STEPS+1)))
        for j, result in enumerate(results):
            scaled_array_copy[j,:,i] = result
    
    return scaled_array_copy

#extracts sequences from class 1
def class1_extractions(ad_class, features):
    '''
    Parameters
    ----------
    Takes features and the classes of each of those features as input
        
    Returns 
    -------
    list with indices of class 1 in original data
    list of features that are ad positive
    
    '''
    class1_features = []
    class1_idx = []
    for i in range(len(features)):
        if ad_class[i] == 1:
            class1_idx.append(i)
            class1_features.append(features[i])
    class1_features = np.array(class1_features)
    return class1_features, class1_idx
