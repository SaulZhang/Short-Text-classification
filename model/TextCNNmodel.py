# -*- coding: utf-8 -*-
import keras
from keras.layers import Dense, Input, Flatten
from keras.layers import Convolution1D,Conv1D, LSTM,MaxPooling1D, Embedding,Dropout,MaxPool1D,GlobalAveragePooling1D,Bidirectional,GRU,BatchNormalization,Activation
from keras.models import Model
from keras.optimizers import *
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from weightnorm import AdamWithWeightnorm
import logging
import numpy as np
import os


train_dataset_file = "./data/train_dataset_generator.txt"

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


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
    print(modelname + 'Build model...')
    sentence = Input(shape = (None, ), dtype = 'float64')
    embedding_layer = Embedding(max_token+1,
                                output_dim=embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)
    # print("sentence:",sentence)
    sentence_embedding = embedding_layer(sentence)
    drp = Dropout(0.2)(sentence_embedding)
    block1 = Convolution1D(128, 1, padding='same')(drp)

    conv2_1 = Convolution1D(256, 1, padding='same')(drp)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    block2 = Convolution1D(128, 3, padding='same')(relu2_1)


    conv3_1 = Convolution1D(256, 3, padding='same')(drp)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    block3 = Convolution1D(128, 5, padding='same')(relu3_1)

    block4 = Convolution1D(128, 3, padding='same')(drp)

    inception = concatenate([block1, block2, block3, block4], axis=-1)

    flat = Flatten()(inception)
    fc = Dense(128)(flat)
    drop = Dropout(0.5)(fc)
    relu = Activation('relu')(drop)
    output = Dense(num_classes, activation='sigmoid')(relu)
    model = Model(inputs = sentence, outputs = output)
    model.compile(optimizer=AdamWithWeightnorm(), loss='categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists('./model490w/TextCNN.h5'):
        print("loading model TextCNN.h5 ... ")
        model.load_weights('./model490w/TextCNN.h5')

    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=modelpath + modelname+ '.h5', monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True,mode='max',period=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./log490w/{}/'.format(modelname), histogram_freq=0, write_graph=True, write_images=True)
    callback_lists=[tensorboard,checkpoint,early_stopping]

    hist = model.fit_generator(generate_arrays_from_file(train_dataset_file,batch_size=batch_size), validation_data=(x_test, to_categorical(y_test,num_classes)),
                     steps_per_epoch = int(4900000/batch_size) ,nb_epoch=epochs, callbacks=callback_lists)

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