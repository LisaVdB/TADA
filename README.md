# TADA
Transcriptional activation domain prediction using deep learning.

## Table of content
* [Architecture](#Architecture)
* [Installation](#Installation)
* [Publications](#Publications)

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

## Publications
S. Mahatma, L. Van den Broeck, N. Morffy, M. V. Staller, L. C. Strader and R. Sozzani, "Prediction and functional characterization of transcriptional activation domains," 2023 57th Annual Conference on Information Sciences and Systems (CISS), Baltimore, MD, USA, 2023, pp. 1-6, doi: 10.1109/CISS56502.2023.10089768.

