'''
TextRNN 模型，包括SimpleRNN/BiLSTM/BiGRU三种模型
'''
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,GRU,SimpleRNN
from keras.layers import  BatchNormalization
from weightnorm import AdamWithWeightnorm
import logging
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import keras
import os
import numpy as np


train_dataset_file = "./data2/train_dataset_generator.txt"


class LSTMpeephole(LSTM):
    def __init__(self, **kwargs):
        super(LSTMpeephole, self).__init__(**kwargs)

    def build(self):
        super(LSTMpeephole, self).build()
        self.P_i = self.inner_init((self.output_dim, self.output_dim))
        self.P_f = self.inner_init((self.output_dim, self.output_dim))
        self.P_c = self.inner_init((self.output_dim, self.output_dim))
        self.P_o = self.inner_init((self.output_dim, self.output_dim))
        self.trainable_weights += [self.P_i, self.P_f, self.P_o]

    def step(self, x, states):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i) + K.dot(c_tm1, self.P_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f) + K.dot(c_tm1, self.P_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c) + K.dot(c_tm1, self.P_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o) + K.dot(c_tm1, self.P_o))
        h = o * self.activation(c)
        return h, [h, c]



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


def train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname,num_classes):
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)
    print(modelname + 'Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(SimpleRNN(128, activation="relu"))
    # model.add(LSTM(128))
    # model.add(Bidirectional(LSTM(200))) ### 输出维度64 GRU
    # model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    # lstm常选参数model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # a stateful LSTM model
    # lahead: the input sequence length that the LSTM
    # https://github.com/keras-team/keras/blob/master/examples/lstm_stateful.py
    # model = Sequential()
    # model.add(LSTM(20,input_shape=(lahead, 1),
    #               batch_size=batch_size,
    #               stateful=stateful))
    # model.add(Dense(num_classes))
    # model.compile(loss='mse', optimizer='adam')

    # model.load_weights('./model490w/TextBiLSTM.h5')

    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=modelpath + modelname+ '.h5', monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True,mode='max',period=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./log/{}/'.format(modelname), histogram_freq=0, write_graph=True, write_images=True)
    callback_lists=[tensorboard,checkpoint,early_stopping]

    hist = model.fit(x_train, to_categorical(y_train,num_classes), validation_data=(x_test, to_categorical(y_test,num_classes)),
                     epochs=epochs, batch_size=batch_size, callbacks=callback_lists)
    max_val_acc = max(hist.history['val_acc'])
    os.rename(modelpath + modelname+ '.h5',modelpath + modelname+'('+str(max_val_acc)+')'+ '.h5')
    
    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    # model.save_weights(modelpath + modelname + '.h5')

def train2(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname,num_classes):
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)
    print(modelname + 'Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(SimpleRNN(128, activation="relu"))
    # model.add(LSTM(128))
    model.add(Bidirectional(LSTMpeephole(units=256)))
    model.add(Dense(num_classes, activation='sigmoid'))
    '''
    from weightnorm import SGDWithWeightnorm
    sgd_wn = SGDWithWeightnorm(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd_wn,metrics=['accuracy'])
    '''
    # try using different optimizers and different optimizer configs
    model.compile(optimizer=AdamWithWeightnorm(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('./model490w/TextBiLSTM-weightnorm(0.9156999999237061).h5')
    # lstm常选参数model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # a stateful LSTM model
    # lahead: the input sequence length that the LSTM
    # https://github.com/keras-team/keras/blob/master/examples/lstm_stateful.py
    # model = Sequential()
    # model.add(LSTM(20,input_shape=(lahead, 1),
    #               batch_size=batch_size,
    #               stateful=stateful))
    # model.add(Dense(num_classes))
    # model.compile(loss='mse', optimizer='adam')

    # patience经过几个epoch后loss不在变化停止训练
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=modelpath + modelname+ '.h5', monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True,mode='max',period=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./log490w/{}/'.format(modelname), histogram_freq=0, write_graph=True, write_images=True)
    callback_lists=[tensorboard,checkpoint,early_stopping]

    hist = model.fit_generator(generate_arrays_from_file(train_dataset_file,batch_size=batch_size), validation_data=(x_test, to_categorical(y_test,num_classes)),
                     steps_per_epoch = int(4900000/batch_size) ,epochs=epochs, callbacks=callback_lists)
    max_val_acc = max(hist.history['val_acc'])
    os.rename(modelpath + modelname+ '.h5',modelpath + modelname+'('+str(max_val_acc)+')'+ '.h5')
    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    # model.save_weights(modelpath + modelname + '.h5')

def train3(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname,num_classes):
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)
    print(modelname+'Build model...')
    model = Sequential()
    model.add(embedding_layer)
    # model.add(SimpleRNN(128, activation="relu"))
    # model.add(LSTM(128))
    # model.add(Bidirectional(LSTM(200))) ### 输出维度64 GRU
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    # lstm常选参数model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # a stateful LSTM model
    # lahead: the input sequence length that the LSTM
    # https://github.com/keras-team/keras/blob/master/examples/lstm_stateful.py
    # model = Sequential()
    # model.add(LSTM(20,input_shape=(lahead, 1),
    #               batch_size=batch_size,
    #               stateful=stateful))
    # model.add(Dense(num_classes))
    # model.compile(loss='mse', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=modelpath + modelname+ '.h5', monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True,mode='max',period=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./log/{}/'.format(modelname), histogram_freq=0, write_graph=True, write_images=True)
    callback_lists=[tensorboard,checkpoint,early_stopping]

    hist = model.fit(x_train, to_categorical(y_train,num_classes), validation_data=(x_test, to_categorical(y_test,num_classes)),
                     epochs=epochs, batch_size=batch_size, callbacks=callback_lists)
    max_val_acc = max(hist.history['val_acc'])
    os.rename(modelpath + modelname+ '.h5',modelpath + modelname+'('+str(max_val_acc)+')'+ '.h5')
    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    # model.save_weights(modelpath + modelname + '.h5')