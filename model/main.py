# -*- coding: utf-8 -*-
"""
主函数，调用各个模型进行训练
"""
from dataPreprocess import *
import TextCNNmodel
import TextRNNmodel
import TextRCNNmodel
import LSTMAttention
import MyModel
import AttentionWithPosition

from keras.layers import Dense, Input, Flatten,Permute,Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import merge
from keras.models import Model
from keras import backend as K
from keras.layers.core import Lambda,RepeatVector
import logging
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import ensemble_weight_average

print("设置参数")

logpath='./model490w/mylog.txt' #日志记录地址
modelpath='./model490w/' #模型保存目录

batch_size = 1024
epochs = 100


embedding_dims = 300
x_train_dataset_pkl = './data/x_train_dataset.pkl'
y_train_dataset_pkl = './data/y_train_dataset.pkl'
x_test_dataset_pkl = './data/x_test_dataset.pkl'
y_test_dataset_pkl = './data/y_test_dataset.pkl'
other_info = './data/other_info.pkl'

print("获取数据")
x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix = getdata(path,embedding_dims,w2vpath,stop_word_filename,word2idx_pkl,label2idx_pkl,x_train_dataset_pkl,y_train_dataset_pkl,x_test_dataset_pkl,y_test_dataset_pkl,other_info)
print(len(x_train))


# TextRNNmodel.train(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"TextSimpleRNN",num_classes=1259)
# TextRNNmodel.train2(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"TextBiLSTM-weightnorm",num_classes=1259)
# TextRNNmodel.train3(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"TextBiGRU",num_classes=1259)
# TextCNNmodel.train(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"TextCNN",num_classes=1259)
# TextAttention.train(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"TextAttention",num_classes=1259)
# TextRCNNmodel.train(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"TextRCNN",hidden_dim_1,hidden_dim_2,num_classes=1259)
# LSTMAttention.train(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"LSTMAttention-weightnorm-512",num_classes=1259)
# MyModel.train(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"MyConBiGRU",num_classes=1259)
# AttentionWithPosition.train(x_train, y_train, x_test, y_test,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,"Attention-wight-norm-WithPositionEmbedding",num_classes=1259)
ensemble_weight_average.modify(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,modelpath=modelpath,epochs=epochs,batch_size=batch_size,logpath = './ensemble_log/mylog.txt',modelname='ensemble-model-avg-weighted')
