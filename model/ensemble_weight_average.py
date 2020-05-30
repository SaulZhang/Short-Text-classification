# -*- coding: utf-8 -*-
'''
采用加权平均的方式进行模型融合
'''
from keras.layers import Dense, Input, Flatten,Permute,Reshape,Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed,Bidirectional
from keras.layers import merge,Activation,Convolution1D, MaxPool1D, GlobalAveragePooling1D,BatchNormalization,PReLU
from keras.layers.merge import concatenate
from keras.models import Model,Sequential
from keras.layers.core import Lambda,RepeatVector
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras import initializers
from keras import constraints
from keras import regularizers
from keras.engine.topology import Layer
from keras.models import load_model
from keras.utils.vis_utils import plot_model

import tensorflow as tf
import keras.losses
import logging
import pandas as pd
import numpy as np
import os
import time
import LSTMAttention
from dataPreprocess import *
import keras
import pickle
import time


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

class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #卤脴脨毛脦陋脜录脢媒
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)
        
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32' ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 #K.arange虏禄脰搂鲁脰卤盲鲁陇拢卢脰禄潞脙脫脙脮芒脰脰路陆路篓脡煤鲁脡
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)

class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1) #脭脷脛鲁脪禄脰赂露篓脰谩拢卢录脝脣茫脮脜脕驴脰脨碌脛脰碌碌脛脌脹录脫潞脥隆拢
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        #脠莽鹿没脰禄麓芦脠毛Q_seq,K_seq,V_seq拢卢脛脟脙麓戮脥虏禄脳枚Mask
        #脠莽鹿没脥卢脢卤麓芦脠毛Q_seq,K_seq,V_seq,Q_len,V_len拢卢脛脟脙麓露脭露脿脫脿虏驴路脰脳枚Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #露脭Q隆垄K隆垄V脳枚脧脽脨脭卤盲禄禄
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #录脝脣茫脛脷禄媒拢卢脠禄潞贸mask拢卢脠禄潞贸softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #脢盲鲁枚虏垄mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def focal_loss(gamma=2.,alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def GRUAttentionModel(sentence_input,maxlen,max_token,embedding_matrix,embedding_dims,num_classes=1259):

    embedding_layer = Embedding(max_token + 1,
                        embedding_dims,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=True)

    # sentence_input = Input(shape=(maxlen,), dtype='float64')
    embedded_sequences = embedding_layer(sentence_input)

    embedded_sequences = Dropout(0.25)(embedded_sequences)

    # embed = Embedding(len(vocab) + 1,300, input_length = 20)(inputs)
    lstm = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embedded_sequences)
    attention = AttLayer()(lstm)
    output = Dense(num_classes, activation='sigmoid')(attention)
    model = Model(sentence_input, output)

    return model

def BiLSTMModel(sentence_input,maxlen,max_token,embedding_matrix,embedding_dims,num_classes=1259):
    embedding_layer = Embedding(max_token + 1,
                            embedding_dims,
                            input_length=maxlen,
                            weights=[embedding_matrix],
                            trainable=True)
    # embedded_sequences = embedding_layer(sentence_input)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200)))
    model.add(Dense(num_classes, activation='sigmoid'))
    # model.compile(optimizer=AdamWithWeightnorm(), loss='categorical_crossentropy', metrics=['accuracy'])
    # drop = Dropout(0.2)(embedded_sequences)
    # lstm = (Bidirectional(LSTM(200)))(drop)
    # output = Dense(num_classes, activation='sigmoid')(drop)
    # model = Model(sentence_input,output)
    return model

def AttentionWithPosition(sentence_input,maxlen,max_token,embedding_matrix,embedding_dims,num_classes=1259):
    embedding_layer = Embedding(max_token + 1,
                            embedding_dims,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=True)

    sentence_input = Input(shape=(maxlen,), dtype='float64')
    embedded_sequences = embedding_layer(sentence_input)
    embedded_sequences = Dropout(0.25)(embedded_sequences)
    # lstm = Bidirectional(LSTM(256))(embedded_sequences)
    # embedded_sequences = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embedded_sequences)
    # embedded_sequences = Dropout(0.2)(embedded_sequences)
    embedded_sequences = Position_Embedding()(embedded_sequences)
    O_seq = Attention(8,16)([embedded_sequences,embedded_sequences,embedded_sequences])
    O_seq = GlobalAveragePooling1D()(O_seq)
    # O_seq = Dropout(0.2)(O_seq)

    # outputs = Dense(1, activation='sigmoid')(O_seq)
    output = Dense(num_classes, activation='sigmoid')(O_seq)

    # model = Model(inputs=S_inputs, outputs=outputs)
    model = Model(sentence_input, output)

    return model

#grid search the best weight for each model
def modify(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,epochs=100,batch_size=128,modelpath='./model/',logpath = './ensemble_log/mylog.txt',modelname='ensemble-model'):#这里修改模型
    maxlen = maxlen[0]
    max_token = max_token[0]
    embedding_matrix = embedding_matrix[0]
    embedding_dims = 300
    num_classes = 1259
    inp = Input(shape=(maxlen,), dtype='float64')#融合主要就是Input是同样的，所以重新建立模型
    model_GRUAttention = GRUAttentionModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)
    model_BiLSTM = BiLSTMModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)
    model_AttentionWithPosition = AttentionWithPosition(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)

    model_GRUAttention.load_weights('C:/Users/Jet Zhang/Desktop/A01-text-classfication/model/GRUAttention(0.9175799998474121).h5')#{'focal_loss_fixed':focal_loss}
    model_BiLSTM.load_weights('C:/Users/Jet Zhang/Desktop/A01-text-classfication/model/TextBiLSTM-weightnorm(0.9156999999237061).h5')
    model_AttentionWithPosition.load_weights('C:/Users/Jet Zhang/Desktop/A01-text-classfication/model/Attention-wight-norm-WithPositionEmbedding(0.9088).h5')
    print("Loading successfully")
    t1 = time.time()
    x_test = x_test
    y_test = y_test
    predict1 = model_GRUAttention.predict(x_test)
    predict1 = np.array(predict1)

    predict2 = model_BiLSTM.predict(x_test)
    predict2 = np.array(predict2)

    predict3 = model_AttentionWithPosition.predict(x_test)
    predict3 = np.array(predict3)

    target = np.array(y_test)#

    best_w1 = 0
    best_w2 = 0
    best_w3 = 0
    best_acc = 0

    for i in range(0,100+1):
        for j in range(0,100+1-i):
            k = 100 - i -j
            predict = (0.28*predict1+0.67*predict2+0.5*predict3)
            predict= np.argmax(predict,axis=1)
            precision = np.mean(target == predict)
            print("With the weight i={} j={} k={},the val_acc is {}".format(i,j,k,precision))
            if precision>best_acc:
                best_w1 = i
                best_w2 = j
                best_w3 = k
                best_acc = precision

    t2 = time.time()
    total_time = t2-t1
    print('the best test acc is:',best_acc," the time spend is:",total_time,"With the weight i={} j={} k={}".format(best_w1,best_w2,best_w3))

