# -*- coding: utf-8 -*-
"""
1、自定义模型 Conv-BiGRU 卷积和循环并行
2、自定义模型 卷积和循环串行
该模型不稳定，可能会出现梯度爆炸
"""
from keras.layers import Dense, Input, Flatten,concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.layers import Convolution1D,Conv1D, MaxPooling1D, Embedding,Dropout,MaxPool1D,GlobalAveragePooling1D,Bidirectional,GRU
from keras.models import Model
import logging
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import keras
import os

def train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname,num_classes):
    sentence = Input(shape=(None,), dtype="float64")
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)
    sentence_embedding = embedding_layer(sentence)
    # c2 = Conv1D(2, 2, activation='relu')(sentence_embedding)
    # p2 = MaxPooling1D(27)(c2)
    # p2 = Flatten()(p2)

    # c3 = Conv1D(2, 3, activation='relu')(sentence_embedding)
    # p3 = MaxPooling1D(26)(c3)
    # p3 = Flatten()(p3)

    # c4 = Conv1D(2, 4, activation='relu')(sentence_embedding)
    # p4 = MaxPooling1D(25)(c4)
    # p4 = Flatten()(p4)

    cnn1 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(sentence_embedding)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn1 = Flatten()(cnn1)

    cnn2 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(sentence_embedding)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn2 = Flatten()(cnn2)

    cnn3 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(sentence_embedding)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn3 = Flatten()(cnn3)

    g1 = Bidirectional(LSTM(128))(sentence_embedding)

    x = concatenate([cnn1, cnn2, cnn3, g1])
    output = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=sentence, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=modelpath + modelname+ '.h5', monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True,mode='max',period=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./log/{}/'.format(modelname), histogram_freq=0, write_graph=True, write_images=True)
    callback_lists=[tensorboard,checkpoint,early_stopping]

    hist = model.fit(x_train, to_categorical(y_train,num_classes), validation_data=(x_test, to_categorical(y_test,num_classes)),
                     epochs=epochs, batch_size=batch_size, callbacks=callback_lists)
    max_val_acc = max(hist.history['val_acc'])
    os.rename(modelpath + modelname+ '.h5',modelpath + modelname+'('+str(max_val_acc)+')'+ '.h5')
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    # model.save(modelpath + modelname + '.h5')