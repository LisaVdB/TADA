# TADA
Transcriptional activation domain prediction using deep learning.

## Table of content
* [Update](#Update)
* [Architecture](#Architecture)
* [Installation](#Installation)
* [Approach](#Approach)
* [Publications](#Publications)

## Update
This repository maintains TADA as it was released for the manuscript "Identification of plant transcriptional activation domains" by Morffy et al. An updated version has been released and is available at https://github.com/ryanemenecker/TADA_T2. This repository also contains a Google Colab implementation for users that wish to generate TADA predictions without installing the TADA model loclly.

## Architecture
To classify protein sequence as an AD, we used a neural network architecture that contains: 
1. Two convolutional neural network layers to extract and compress sequence information from the input
2. An attention layer to selectively focus on the features that are more important for the prediction
3. Two bidirectional long short-term memory (biLSTM) layers to capture the interdependence of the sub-sequences in a sequence
4. A dense layer to connect to the output layer


## Installation

Set up the following environment
* python 3.10.6
* tensorflow 2.10.0
* scikit-learn 1.2.2
* alphapredict 1.0 and protfasta as dependency
* localcider 0.1.19
* pandas 2.0.0
* openTSNE 0.7.1
* shap 0.41.0
* seaborn 0.12.2
* biopython

```
conda create -n tada python=3.10.6
conda activate tada
pip install tensorflow==2.10.0
pip install scikit-learn==1.2.2
pip install pandas==2.0.0
pip install localcider
pip install protfasta
pip install alphaPredict
conda install --channel conda-forge opentsne
pip install shap
pip install seaborn
pip install biopython
```

TADA has been tested on Windows 10.

## Approach
### Preprocessing and model training
All fragments from our experimental datasets (TrainingsData.csv) were preprocessed and split into training, validation, and test set in the Save_train-test-split.py script, which uses custom build functions from the Preprocessing.py script. TADA was trained using the Training.py script. The performance metrics for the test set were presented in the manuscript. Fragments were split into a 90%-10% train-validation proportion to retrain TADA for final predictions. Tiles created for the training datasets as well as prediction datasets were generated using the Tiling.py script, which inputs a fasta file and outputs the 40-aa tiles in a .csv file.

### Making predictions
To make predictions with TADA, use the Predictions.py script. In case the sequences are longer than 40 amino acids, use the split_seq() function from the Preprocessing.py script. Depending on the number of sequences to test, predictions can take from a few minutes to an hour on a standard desktop computer.

### SHAP analysis and clustering
To rank the features according to impact to predict ADs, use the SHAP.py script. To identify AD subtypes by clustering the AD fragments, use the Clustering.py script. The clustering of the AD fragments depends on the computed features and SHAP values. 

## Publications
S. Mahatma*, L. Van den Broeck*, N. Morffy, M. V. Staller, L. C. Strader and R. Sozzani, "Prediction and functional characterization of transcriptional activation domains," 2023 57th Annual Conference on Information Sciences and Systems (CISS), Baltimore, MD, USA, 2023, pp. 1-6, doi: 10.1109/CISS56502.2023.10089768.

