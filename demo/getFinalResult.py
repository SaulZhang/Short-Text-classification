'''
Date:2019-3-24
Author:SaulZhang
Description:对50w以及450w数据打标签得到最终提交的文件
'''
from flask import Flask, request, render_template
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
import pandas as pd
import time
import jieba
import pickle,pprint
import re
import string
import sys
import xlrd
import os
import tensorflow as tf
import numpy as np


other_info_pkl = './data/other_info.pkl'
word2idx_pkl =  './data/word2idx_dict.pkl'

other_info_dataset = open(other_info_pkl, 'rb')
other_info_data = pickle.load(other_info_dataset)
maxlen = other_info_data[0]
max_token = other_info_data[1]
embedding_matrix = other_info_data[2]

label2idx = './data/label2idx_dict.pkl'
label2idx_pkl_file = open(label2idx, 'rb')
label_dict = pickle.load(label2idx_pkl_file)
model_path = './model/TextBiLSTM-weightnorm(0.9156999999237061).h5'

print("Initialization and Loading the model...")

maxlen = maxlen[0]
max_token = max_token[0]
embedding_matrix = embedding_matrix[0]
embedding_dims = 300


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

def BiLSTMModel(sentence_input,maxlen,max_token,embedding_matrix,embedding_dims,num_classes=1259):
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200)))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model


global graph
graph = tf.get_default_graph()
inp = Input(shape=(maxlen,), dtype='float64')

# model_BiLSTM = BiLSTMModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=embedding_dims)
# model_BiLSTM.load_weights(model_path)
model_GRUAttention = GRUAttentionModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)
model_BiLSTM = BiLSTMModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)
model_AttentionWithPosition = AttentionWithPosition(inp,maxlen,max_token,embedding_matrix,embedding_dims=300)
model_GRUAttention.load_weights('./model/GRUAttention(0.9175799998474121).h5')#{'focal_loss_fixed':focal_loss}
model_BiLSTM.load_weights('./model/TextBiLSTM-weightnorm(0.9156999999237061).h5')
model_AttentionWithPosition.load_weights('./model/Attention-wight-norm-WithPositionEmbedding(0.9088).h5')


#将多个句子转变为word2idx的形式
def processMultiExample(testfilename,word2idxfilename,maxlen):
    print('Loading multi-data...')
    f_read = open(testfilename,encoding='utf-8')
    # d = pd.read_csv(testfilename,encoding='utf-8',sep = '\t')
    # d.columns=['title']
    # d=d[-pd.isnull(d["title"])].reset_index(drop=True)
    dx=[]
    word_to_id_dataset = open(word2idxfilename, 'rb')
    word_to_id = pickle.load(word_to_id_dataset)
    f_read.readline()
    for idx,line in enumerate(f_read):
        if idx%50000==0:
            print("process ",idx," line")
        line = line.strip('\n')
        line = line.translate(str.maketrans('', '',"★[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+" ))
        seg_list = jieba.cut(line,cut_all=False)
        seg_list = list(seg_list)
        linelist = []
        for seg in seg_list:
            if (seg).isdigit() == True:
                continue
            if seg in word_to_id.keys():
                linelist.append(word_to_id[seg])
        dx.append(linelist)
    x_test= dx[:]
    x_test = pad_sequences(x_test, maxlen=maxlen)
    return x_test

def predictBatchExample(file_path,x_test):#这里修改模型
    new_label_dict = {v : k for k, v in label_dict.items()}#将类别的编号转为idx
    
    file_result_path = "./Finalresult/train-result."+file_path.split(".")[-1]
    f_w = open(file_result_path,'w',encoding='utf-8')

    f_read = open(file_path,encoding='utf-8')
    f_read.readline()
    all_text=[i.strip() for i in f_read]
    # d = pd.read_csv(file_path,encoding='utf-8',sep = '\t')#重新读取文件的每一行,便于将其文本与标签存储到写出的文件当中
    # d.columns=['title']
    # d1=d[-pd.isnull(d["title"])].reset_index(drop=True)
    # d2=d[-pd.isnull(d["label"])].reset_index(drop=True)
    f_w.write("ITEM_NAME"+'\t'+"TYPE"+'\n')

    with graph.as_default():
        for idx,(line,test_data) in enumerate(zip(all_text,x_test)):
            if idx%50000==0:
                print("classify ",idx," line")

            test_data = np.array(test_data)
            test_data = test_data.reshape(1,21)

            predict1 = model_GRUAttention.predict(test_data)
            predict1 = np.array(predict1)

            predict2 = model_BiLSTM.predict(test_data)
            predict2 = np.array(predict2)

            predict3 = model_AttentionWithPosition.predict(test_data)
            predict3 = np.array(predict3)

            predict = (0.28*predict1+0.67*predict2+0.05*predict3)

            pre= np.argmax(predict,axis=1)

            f_w.write(line+'\t'+new_label_dict[pre[0]]+'\n')
    f_w.close()


def main():
    file_path = "./testdata/test-100example.tsv"
    t1 = time.time()
    x_test = processMultiExample(file_path,word2idx_pkl,maxlen)
    print("文件处理完毕")
    batchResult = predictBatchExample(x_test=x_test,file_path=file_path)
    t2 = time.time()

    # file_result_path = "./Finalresult/result-test."+file_path.split(".")[-1]
    # f_w = open(file_result_path,'w',encoding='utf-8')
    # d = pd.read_csv(file_path,encoding='utf-8',sep = '\t')#重新读取文件的每一行,便于将其文本与标签存储到写出的文件当中
    # d.columns=['title']
    # d1=d[-pd.isnull(d["title"])].reset_index(drop=True)
    # d2=d[-pd.isnull(d["label"])].reset_index(drop=True)
    # f_w.write("ITEM_NAME"+'\t'+"TYPE"+'\n')
    # for (line,pred) in zip(d1["title"],batchResult):
        # print(line,'\t',label)
        # f_w.write(line+'\t'+pred+'\n')
    # f_w.close()

    print("预测完毕")
    print("一共耗时{}秒".format(t2-t1))

if __name__ == '__main__':
    main()