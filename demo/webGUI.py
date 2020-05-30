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
import predict
import tensorflow as tf
import numpy as np

__author__ = 'zzh'

UPLOAD_FOLDER = './testdata'
ALLOWED_EXTENSIONS = set(['txt,xlsx,tsv,xls'])
app = Flask(__name__)
# app._static_folder = UPLOAD_FOLDER

other_info_pkl = './data/other_info.pkl'
word2idx_pkl =  './data/word2idx_dict.pkl'

other_info_dataset = open(other_info_pkl, 'rb')
other_info_data = pickle.load(other_info_dataset)
maxlen = other_info_data[0]
max_token = other_info_data[1]
embedding_matrix = other_info_data[2]

testfilename = './data/test_pre.tsv'
label2idx = './data/label2idx_dict.pkl'
label2idx_pkl_file = open(label2idx, 'rb')
label_dict = pickle.load(label2idx_pkl_file)
model_path = './model/TextBiLSTM-weightnorm(0.9156999999237061).h5'

print("Initialization and Loading the model...")

maxlen = maxlen[0]
max_token = max_token[0]
embedding_matrix = embedding_matrix[0]
embedding_dims = 300

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
model_BiLSTM = BiLSTMModel(inp,maxlen,max_token,embedding_matrix,embedding_dims=embedding_dims)
model_BiLSTM.load_weights(model_path)


def predictSingleExample(x_test):#这里修改模型
    
    # label2idx = './data/label2idx_dict.pkl'

    new_label_dict = {v : k for k, v in label_dict.items()}#将类别的编号转为idx
    #融合主要就是Input是同样的，所以重新建立模型
    t1 = time.time()
    with graph.as_default():
        predict = model_BiLSTM.predict(x_test)#
    predict = np.array(predict)
    predict= np.argmax(predict,axis=1)

    return new_label_dict[predict[0]]


def predictBatchExample(x_test):#这里修改模型
    new_label_dict = {v : k for k, v in label_dict.items()}#将类别的编号转为idx
    t1 = time.time()
    with graph.as_default():
        predict = model_BiLSTM.predict(x_test)
    predict = np.array(predict)
    predict= np.argmax(predict,axis=1)
    result =[]
    for pre in predict:
        result.append(new_label_dict[pre])

    return result


def allowed_files(filename):
    if '.' in filename and filename.rsplit('.', 1)[1] == 'txt':
        return 1
    else:
        # if '.' in filename and filename.rsplit('.', 1)[1] == 'xlsx' or filename.rsplit('.', 1)[1] == 'xls':
        #     return 2
        # else:
        if '.' in filename and filename.rsplit('.', 1)[1] == 'tsv':
            return 3
        else:
            return 4


@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html', error='')


@app.route("/help", methods=['POST', 'GET'])
def Use_Assistant():
    if request.method == 'POST':
        return render_template('help.html')
    else:
        return render_template('help.html')


@app.route("/data_processing", methods=['POST'])
def data_processing():
    if request.method == 'POST' and request.form["data_processing_btn"] == "批量数据分类":
            if 'file' not in request.files:
                return render_template('index.html', error='batch')
            else:
                file = request.files['file']
                # print("file:",file)
                # print("file.filename:",file.filename)

                
                old_file_name = file.filename

                if file and allowed_files(old_file_name) == 1:#处理txt格式
                    filename = old_file_name
                    file.save(os.path.join(UPLOAD_FOLDER, filename))
                    file_path = os.path.join(UPLOAD_FOLDER, filename)

                    x_test = predict.processMultiExample(file_path,word2idx_pkl,maxlen)
                    print("开始预测")
                    t1 = time.time()

                    #异常处理
                    if x_test=="FILE_FORMAT_ERROR":
                        print("FILE_FORMAT_ERROR")
                        return render_template('index.html', error='wrong_format')

                    batchResult = predictBatchExample(x_test=x_test)
                    t2 = time.time()
                    file_result_path = "./result/result-"+old_file_name.split(".")[0]+'.'+old_file_name.split(".")[-1]
                    f_w = open(file_result_path,'w')
                    
                    d = pd.read_csv(file_path,encoding='gb18030',sep = '\t')#重新读取文件的每一行,便于将其文本与标签存储到写出的文件当中
                    d.columns=['title']
                    d=d[-pd.isnull(d["title"])].reset_index(drop=True)

                    f_w.write("ITEM_NAME"+'\t'+"TYPE"+'\n')
                    for (line,label) in zip(d["title"],batchResult):
                        # print(line,'\t',label)
                        f_w.write(line+'\t'+label+'\n')
                    f_w.close()
                    print("预测完毕")
                    print("一共耗时{}秒".format(t2-t1))
                    file_path = file_result_path
                    text_list = []
                    label_list_1 = []
                    label_list_2 = []
                    label_list_3 = []
                    f=  open(file_path, 'r')
                    lines = 0
                    f.readline()
                    for idx,line in enumerate(f):
                        text,label = line.strip().split('\t')
                        text_list.append(text)
                        label_list_1.append(label.split("--")[0])
                        label_list_2.append(label.split("--")[1])
                        label_list_3.append(label.split("--")[2])

                        # label_list.append(label)
                        lines = idx
                        print("text=",text,"label=",label)
                    if lines <= 200:
                        return render_template('Batch_data_classification_results.html', filename=os.path.abspath(file_path),ITEM_NAME=text_list, TYPE_ONE=label_list_1,TYPE_TWO=label_list_2, TYPE_THREE=label_list_3,LENGTH=lines)
                    else:
                        return render_template('Wait.html', filename=os.path.abspath(file_path),ITEM_NAME=text_list, TYPE_ONE=label_list_1,TYPE_TWO=label_list_2, TYPE_THREE=label_list_3,LENGTH=200)
                else:
                    if file and allowed_files(old_file_name) == 3:#处理tsv格式
                        filename = old_file_name
                        file_path = os.path.join(UPLOAD_FOLDER, filename)
                        print("file_path:",file_path)
                        x_test = predict.processMultiExample(file_path,word2idx_pkl,maxlen)

                        print("开始预测")
                        t1 = time.time()
                        
                        #异常处理
                        if x_test=="FILE_FORMAT_ERROR":
                            print("FILE_FORMAT_ERROR")
                            return render_template('index.html', error='wrong_format')
                        
                        batchResult = predictBatchExample(x_test=x_test)

                        t2 = time.time()
                        file_result_path = "./result/result-"+old_file_name.split(".")[0]+'.'+old_file_name.split(".")[-1]
                        f_w = open(file_result_path,'w')
                        d = pd.read_csv(file_path,encoding='gb18030',sep = '\t')
                        d.columns=['title']
                        d=d[-pd.isnull(d["title"])].reset_index(drop=True)
                        f_w.write("ITEM_NAME"+'\t'+"TYPE"+'\n')
                        for (line,label) in zip(d["title"],batchResult):
                            # print(line,'\t',label)
                            f_w.write(line+'\t'+label+'\n')
                        f_w.close()
                        print("预测完毕")

                        print("一共耗时{}秒".format(t2-t1))
                        file_path = file_result_path
                        # print("file_path:"*100,file_path)
                        text_list = []
                        label_list_1 = [] 
                        label_list_2 = []
                        label_list_3 = []
                        f=  open(file_path, 'r')
                        lines = 0
                        f.readline()
                        for idx,line in enumerate(f):
                            text,label = line.strip().split('\t')
                            text_list.append(text)
                            
                            label_list_1.append(label.split("--")[0])
                            label_list_2.append(label.split("--")[1])
                            label_list_3.append(label.split("--")[2])
                            lines = idx
                            # print("text=",text,"label=",label)
                            
                        if lines <= 200:
                            return render_template('Batch_data_classification_results.html', filename=os.path.abspath(file_path),ITEM_NAME=text_list, TYPE_ONE=label_list_1,TYPE_TWO=label_list_2, TYPE_THREE=label_list_3,LENGTH=lines)
                        else:
                            return render_template('Wait.html', filename=os.path.abspath(file_path),ITEM_NAME=text_list, TYPE_ONE=label_list_1,TYPE_TWO=label_list_2, TYPE_THREE=label_list_3,LENGTH=200)
                    else:
                        return render_template('index.html', error='wrong_format')
    else:
        if request.method == 'POST' and request.form["data_processing_btn"] == "单个数据分类":
            if request.form["text"] == '':
                return render_template('index.html', error='single')
            else:
                item_name = request.form["text"]
                string = item_name
                x_test = predict.processSingleExample(string=string,word2idx=word2idx_pkl,maxlen=maxlen)
                print("开始预测")
                t1 = time.time()
                print("x_test:",x_test)
                singleResult = predictSingleExample(x_test=x_test)
                t2 = time.time()
                print("预测完毕")
                print("预测结果为:",singleResult)
                print("一共耗时{}秒".format(t2-t1)) 
                return render_template('Single_data_classification_result.html', ITEM_NAME=item_name,ID=1, TYPE_ONE=singleResult.split("--")[0], TYPE_TWO=singleResult.split("--")[1], TYPE_THREE=singleResult.split("--")[2])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
