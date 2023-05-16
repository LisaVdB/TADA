import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv1D, Bidirectional, LSTM, Layer
from tensorflow.keras import regularizers
import keras.backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from loss import focal_loss

np.random.seed(1258)  # reproducibility

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
        
    def get_config(self):
        config = super().get_config()
        return config

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1), initializer="normal")
        super(attention,self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

def create_model(SHAPE, kernel_size=2, filters=100, activation_function = 'gelu', learning_rate=1e-3, dropout=0.3, bilstm_output_size=100):
    """
    Define the NN architecture
    """
    model = Sequential()
    
    model.add(Conv1D(filters=filters, 
                     kernel_size=kernel_size,
                     padding='valid',
                     activation=activation_function,
                     strides=1, input_shape = SHAPE,
                     kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4)
                    )
                    )
    model.add(Dropout(dropout)) 
    model.add(Conv1D(filters=filters,
                 kernel_size=kernel_size,
                 padding = 'valid',
                 activation=activation_function,
                 strides=1))
    model.add(Dropout(dropout))
    
    model.add(attention())
    
    model.add(Bidirectional(LSTM(bilstm_output_size, return_sequences=True))) # Creates Long short term memory RNN and applies a bidirectional wrapper on it
    model.add(Bidirectional(LSTM(bilstm_output_size))) 
    model.add(Dense(2, activation="softmax")) 
    opt = Adam(learning_rate=learning_rate)
    loss_function = focal_loss(alpha = 0.45)
    
    model.compile(loss = loss_function, 
                  optimizer = opt,
                  metrics = metric)
    return model

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val 

metric = [tf.keras.metrics.Precision(name = 'precision'),
tf.keras.metrics.Recall(name = 'recall'), 
tf.keras.metrics.AUC(name = 'auc', curve = 'ROC'),
tf.keras.metrics.CategoricalAccuracy(name ='accuracy'),
tf.keras.metrics.AUC(name = 'aupr', curve = 'PR'),
f1_metric
]

def plot_metric(training_metric, validation_metric, label):
    plt.figure()
    plt.plot(np.arange(1, len(training_metric) + 1), training_metric, label='train', color = "blue")
    plt.plot(np.arange(1, len(training_metric) + 1), validation_metric, label='validation', color = "red")
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(label)
    plt.legend()    
    plt.savefig('{0}.png'.format(label))
