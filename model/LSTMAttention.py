'''
Bi-LSTM + Attention模型
'''
# -*- coding: utf-8 -*-
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed,Bidirectional
from keras.layers import merge,Activation
from keras.models import Model,Sequential
from keras import backend as K
from keras.layers.core import Lambda,RepeatVector
from keras.layers import Dense, Input, Flatten,Permute,Reshape
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D
from keras.layers import  BatchNormalization
from keras import initializers
from keras import constraints
from keras import regularizers
from keras.engine.topology import Layer
from keras.models import load_model
from weightnorm import AdamWithWeightnorm
import keras.losses
import random as rand
import pandas as pd
import numpy as np
import os
import time
import keras
import tensorflow as tf
import logging
'''
Compatible with tensorflow backend
'''
# Hierarchical Model with Attention

train_dataset_file = "./data/train_dataset_generator.txt"

def process_line(line):
    tmp = [int(val) for val in line.strip().split(',')]
    x = np.array(tmp[:-1])
    y = np.array(tmp[-1:])
    return x,y
 
def generate_arrays_from_file(path,batch_size):
    while 1:
        f = open(path)
        cnt = 0
        X =[]
        Y =[]
        for line in f:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(line)
            X.append(x)
            Y.append(y)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                yield (np.array(X), np.array(to_categorical(Y,1259)).reshape(batch_size,1259))
                X = []
                Y = []
    f.close()


class AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None, 
                 bias_regularizer=None, kernel_constraint=None, 
                 bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        
        self.built = True
        
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W) # (x, 40, 1)
        uit = K.squeeze(uit, -1) # (x, 40)
        uit = uit + self.b # (x, 40) + (40,)
        uit = K.tanh(uit) # (x, 40)

        ait = uit * self.u # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait) # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            ait = mask*ait #(x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname,num_classes):
    
    embedding_layer = Embedding(max_token + 1,
                            embedding_dims,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=True)

    sentence_input = Input(shape=(maxlen,), dtype='float64')
    embedded_sequences = embedding_layer(sentence_input)

    embedded_sequences = Dropout(0.25)(embedded_sequences)
    # embed = Embedding(len(vocab) + 1,300, input_length = 20)(inputs)
    lstm = Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embedded_sequences)
    attention = AttLayer()(lstm)
    output = Dense(num_classes, activation='sigmoid')(attention)
    model = Model(sentence_input, output)


    model.compile(optimizer=AdamWithWeightnorm(), loss='categorical_crossentropy', metrics=['accuracy'])
    if os.path.exists('./model490w/GRUAttention-weightnorm-512.h5'):
        print("Loading the model --GRUAttention-weightnorm-512.h5")
        model.load_weights('./model490w/GRUAttention-weightnorm-512.h5')

    early_stopping = EarlyStopping(monitor='acc', patience=10)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=modelpath + modelname+ '.h5', monitor='acc',verbose=1,save_best_only=True,save_weights_only=True,mode='max',period=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logw490/{}/'.format(modelname), histogram_freq=0, write_graph=True, write_images=True)
    callback_lists=[tensorboard,checkpoint,early_stopping]

    hist = model.fit_generator(generate_arrays_from_file(train_dataset_file,batch_size=batch_size), validation_data=(x_test, to_categorical(y_test,num_classes)),
                     steps_per_epoch = int(4900000/batch_size) ,epochs=epochs, callbacks=callback_lists)
    max_val_acc = max(hist.history['val_acc'])
    os.rename(modelpath + modelname+ '.h5',modelpath + modelname+'('+str(max_val_acc)+')'+ '.h5')
    
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)