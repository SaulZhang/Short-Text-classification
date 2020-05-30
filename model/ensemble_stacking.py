# -*- coding: utf-8 -*-
'''
采用stacking机制进行模型融合
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

import tensorflow as tf
import keras.losses
import logging
import pandas as pd
import numpy as np
import os
import time
import h5py
from keras.models import load_model

import LSTMAttention
from dataPreprocess import *
import keras

print("获取数据")
x_train_dataset_pkl = './data1/x_train_dataset.pkl'
y_train_dataset_pkl = './data1/y_train_dataset.pkl'
x_test_dataset_pkl = './data1/x_test_dataset.pkl'
y_test_dataset_pkl = './data1/y_test_dataset.pkl'
other_info = './data1/other_info.pkl'
logpath = './ensemble_log/mylog.txt'
# x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix = getdata('',0,'','','','',x_train_dataset_pkl,y_train_dataset_pkl,x_test_dataset_pkl,y_test_dataset_pkl,other_info)



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

def Mymodel(sentence,maxlen,max_token,embedding_matrix,embedding_dims,num_classes=1259):
    # sentence = Input(shape=(None,), dtype="float64")
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)
    sentence_embedding = embedding_layer(sentence)

    cnn1 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(sentence_embedding)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn1 = Flatten()(cnn1)

    cnn2 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(sentence_embedding)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn2 = Flatten()(cnn2)

    cnn3 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(sentence_embedding)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn3 = Flatten()(cnn3)

    g1 = Bidirectional(GRU(128))(sentence_embedding)

    x = concatenate([cnn1, cnn2, cnn3, g1])
    output = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=sentence, outputs=output)
    return model


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
    lstm = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embedded_sequences)
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
    # print(modelname+'Build model...')
    embedded_sequences = embedding_layer(sentence_input)

    # model.add(SimpleRNN(128, activation="relu"))
    # model.add(LSTM(128))
    # model.add(Bidirectional(LSTM(200))) ### 输出维度64 GRU
    gru = (Bidirectional(LSTM(200)))(embedded_sequences)
    drop = Dropout(0.2)(gru)
    output = Dense(num_classes, activation='sigmoid')(drop)
    model = Model(sentence_input,output)
    return model


def TextCNNModel(sentence_input,maxlen,max_token,embedding_matrix,embedding_dims,num_classes=1259):
    embedding_layer = Embedding(max_token+1,
                                output_dim=embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)
    # print("sentence:",sentence)
    sentence_embedding = embedding_layer(sentence_input)
    cnn = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(sentence_embedding)
    cnn = MaxPool1D(pool_size=4)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(256)(cnn)
    rnn = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(sentence_embedding)
    rnn = Dense(256)(rnn)
    con = concatenate([cnn,rnn], axis=-1)
    output = Dense(num_classes, activation='softmax')(con)
    model = Model(inputs = sentence_input, outputs = output)
    return model


def merge_model(maxlen,max_token,embedding_matrix):

    inp = Input(shape=(maxlen,), dtype='float64')#融合主要就是Input是同样的，所以重新建立模型
    
    model_GRUAttention = GRUAttentionModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)
    model_BiLSTM = BiLSTMModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)
    model_TextCNN = TextCNNModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)

    model_GRUAttention.load_weights('./model1/GRUAttention(0.86002).h5')#{'focal_loss_fixed':focal_loss}
    model_BiLSTM.load_weights('./model1/TextBiLSTM(0.8606).h5')
    model_TextCNN.load_weights('./model1/TextCNN(0.848470000038147).h5')

    # print(model_GRUAttention.shape)
    r1=model_GRUAttention.output#获得输出
    r2=model_BiLSTM.output
    r3=model_TextCNN.output

    x=concatenate([r1,r2,r3],axis=1)#拼接输出，融合成功

    model=Model(inp,x)
    return model


def modify(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,epochs=100,batch_size=128,modelpath='./model/',logpath = './ensemble_log/mylog.txt',modelname='ensemble-model'):#这里修改模型
    maxlen = maxlen[0]
    max_token = max_token[0]
    embedding_matrix = embedding_matrix[0]
    embedding_dims = 300
    num_classes = 1259
    origin_model=merge_model(maxlen,max_token,embedding_matrix)
    for layer in origin_model.layers:
        layer.trainable = False#原来的不训练
        
    inp=origin_model.input
    x=origin_model.output
    
    den=Dense(512,name="fine_dense_ensemble")(x)
    l=PReLU()(den)
    l=Dropout(0.35,name="dropout_ensemble")(l)
    result=Dense(num_classes,activation="softmax",name='dense_ensemble')(l)
    
    model=Model(outputs=result,input=inp)
    #编译model
    adam = keras.optimizers.Adam(lr = 0.0005, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    #adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    #sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)
 
    #reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=modelpath + modelname+ '.h5', monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True,mode='max',period=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./log/{}/'.format(modelname), histogram_freq=0, write_graph=True, write_images=True)
    callback_lists=[tensorboard,checkpoint,early_stopping]

    hist = model.fit(x_test[:len(x_test)//2], to_categorical(y_test[:len(y_test)//2],num_classes), validation_data=(x_test[len(x_test)//2:], to_categorical(y_test[len(y_test)//2:],num_classes)),
                     epochs=epochs, batch_size=batch_size, callbacks=callback_lists)
    max_val_acc = max(hist.history['val_acc'])
    os.rename(modelpath + modelname+ '.h5',modelpath + modelname+'('+str(max_val_acc)+')'+ '.h5')
                     
    #输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)


def main(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix):
    maxlen = maxlen[0]
    max_token = max_token[0]
    embedding_matrix = embedding_matrix[0]
    embedding_dims = 300
    num_classes = 1259
    model = modify(logpath = './ensemble_log/mylog.txt',modelname='ensemble-model')

if __name__ == '__main__':

    main()