# -*- coding: utf-8 -*-
'''
实现数据的预处理
'''
import pandas as pd
import numpy as np
import gensim
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import importlib
import sys
import pickle,pprint
import os

importlib.reload(sys)

"""
description 将训练的数据用pickle进行持久化，方便后面的每一次读取，存储格式为[[x_train, y_train, x_test, y_test],[maxlen],[max_token],[embedding_matrix]]
args:
    path='./data/train.tsv'                      #数据集所在的路径
    embedding_dims = 300  #词向量长度             #词向量的维度
    w2vpath='./w2vmodel/word2vec.model'          #预训练的语言模型的存储路径
    word2idx_pkl = "./data/word2idx_dict.pkl"    #存储分词字典的持久化文件
    label2idx_pkl = './data/label2idx_dict.pkl'  #存储标签字典的持久化文件
    dataset_pkl = './data/dataset.pkl'           #存储最终的训练集和验证集的数据划分及相关信息存储格式为[[x_train, y_train, x_test, y_test],[maxlen],[max_token],[embedding_matrix]]

return :
    x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix
"""

def getdata(path,embedding_dims,w2vpath,stop_word_filename,word2idx_pkl,label2idx_pkl,x_train_dataset_pkl,y_train_dataset_pkl,x_test_dataset_pkl,y_test_dataset_pkl,other_info_pkl):
    
    if os.path.exists(x_train_dataset_pkl) == True:#使用数据持久化模块
        x_train_dataset = open(x_train_dataset_pkl, 'rb')
        x_train_data = pickle.load(x_train_dataset)

        y_train_dataset = open(y_train_dataset_pkl, 'rb')
        y_train_data = pickle.load(y_train_dataset)

        x_test_dataset = open(x_test_dataset_pkl, 'rb')
        x_test_data = pickle.load(x_test_dataset)

        y_test_dataset = open(y_test_dataset_pkl, 'rb')
        y_test_data = pickle.load(y_test_dataset)

        other_info_dataset = open(other_info_pkl, 'rb')
        other_info_data = pickle.load(other_info_dataset)

        # label_dataset = open('./data/label2idx_dict.pkl', 'rb')
        # other_info_data = pickle.load(label_dataset)
        # print(other_info_data)
        # print(len(other_info_data))
        return x_train_data,y_train_data,x_test_data,y_test_data,other_info_data[0],other_info_data[1],other_info_data[2]

    else:
        print('Loading data...')
        file_train = open(path,encoding='utf-8')
        file_test = open("./data2/test-10w.tsv",encoding='utf-8')
        
        train_text = []
        train_label = []
        for idx,line in enumerate(file_train):
            # if idx>10:break
            try:
                text,target = line.split('\t')
                target=target.strip('\n')
                text = text.replace("  ","")
                train_text.append(text)
                train_label.append(target)
            except:
                print(line.strip())

        test_text = []
        test_label = []
        for idx,line in enumerate(file_test):
            # if idx>10:break
            try:
                text,target = line.split('\t')
                target=target.strip('\n')
                text = text.replace("  ","")
                test_text.append(text)
                test_label.append(target)
            except:
                print(line.strip())

        print("开始打乱顺序")
        #随机打乱顺序
        import random
        cc = list(zip(train_text,train_label))
        random.shuffle(cc)
        train_text,train_label=zip(*cc)

        train_text=list(train_text)
        train_label=list(train_label)
        all_data=set()
        print("step-1 ")
        dict_frequency = {}
        for idx,line in enumerate(train_text+test_text):
            # if idx>10:break
            ws=str(line).split(" ")
            for w in ws:
                #     if isStopWord(w,stopWordList) == True or (w).isdigit()==True:#去除停用词和完全是数值的单词
                #         continue
                all_data.add(w)
                if(w not in dict_frequency.keys()):
                    dict_frequency[w]=0
                else:
                    dict_frequency[w]+=1

        words = []
        dict_frequency=sorted(dict_frequency.items(),key = lambda x:x[1],reverse = True)
        x_train_dataset_pkl = './data2/dict_frequency.pkl'
        x_test_dataset_pkl_file = open(x_test_dataset_pkl,'wb')
        pickle.dump(x_train,x_train_dataset_pkl_file)
        # print(len(dict_frequency))
        # print(dict_frequency)
        for i,item in  enumerate(dict_frequency):
            if i > 100000:break
            words.append(item[0])
        
        print(len(words))

        print("step-2")
        word_to_id = dict(zip(words, range(len(words))))
        dx=[]
        for idx,line in enumerate(train_text+test_text):
            # if idx>10:break
            ws=str(line).split(" ")
            dx.append([word_to_id[w] for w in ws if w in word_to_id])
        # dy=list(d['lable'])
        dy=train_label+test_label
        count = 0
        label_dict = {}

        for idx,label in enumerate(dy):
            # if idx>10:break
            if label not in label_dict.keys():
                label_dict[label]= count
                count += 1
        for idx in range(len(dy)):
            # if idx>10:break
            dy[idx] = label_dict[dy[idx]]

        print(label_dict)
        # sys.exit(0)

        print("step-3")
        print('Average  sequence length: {}'.format(np.mean(list(map(len, dx)), dtype=int)))
        # set parameters:
        # maxlen=np.max(list(map(len, dx))) #maxlen = 29  最长文本词数
        
        sortmaxlen=np.sort(list(map(len, dx)), axis=0)
        print(sortmaxlen)
        maxlen = sortmaxlen[int(len(dy)*0.8)]
        print("maxlen:",maxlen)
        num_classes = len(label_dict)
        # categorical_labels = to_categorical(int_labels, num_classes=num_classes)

        inx=int(len(dx)/5*4)

        x_train, y_train, x_test, y_test = dx[0:int(len(dy)*0.98)],dy[0:int(len(dy)*0.98)],dx[int(len(dy)*0.98):],dy[int(len(dy)*0.98):]
        # y_train = to_categorical(y_train,num_classes)
        # y_test = to_categorical(y_test,num_classes)

        # print([x_train, y_train, x_test, y_test])

        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        print('Pad sequences (samples x time)')
        x_train = pad_sequences(x_train, maxlen=maxlen)
        x_test = pad_sequences(x_test, maxlen=maxlen)

        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        # print('y_train shape:', y_train.shape)
        # print('y_test shape:', y_test.shape)
        word2idx_pkl_file = open(word2idx_pkl,'wb')
        pickle.dump(word_to_id,word2idx_pkl_file)

        label2idx_pkl_file = open(label2idx_pkl,'wb')
        pickle.dump(label_dict,label2idx_pkl_file)

        x_train_dataset_pkl_file = open(x_train_dataset_pkl,'wb')
        pickle.dump(x_train,x_train_dataset_pkl_file)
        y_train_dataset_pkl_file = open(y_train_dataset_pkl,'wb')
        pickle.dump(y_train,y_train_dataset_pkl_file)

        x_test_dataset_pkl_file = open(x_test_dataset_pkl,'wb')
        pickle.dump(x_test,x_test_dataset_pkl_file)
        y_test_dataset_pkl_file = open(y_test_dataset_pkl,'wb')
        pickle.dump(y_test,y_test_dataset_pkl_file)

        print('Indexing word vectors.')
        embeddings_index = {}
        model = gensim.models.Word2Vec.load(w2vpath)


        #初始化一个0向量 统计未出现词个数
        null_word=np.zeros(embedding_dims)
        null_word_count=0
        print("step-4")
        for word in words:
            try:
                embeddings_index[word]=model[word]
            except:
                embeddings_index[word]=null_word
                null_word_count+=1
                print("word=,",word)
        print('Found %s word vectors.' % len(embeddings_index))
        print('Found %s null word.' % null_word_count)
        print('Preparing embedding matrix.')
        max_token = len(word_to_id)
        for word, i in word_to_id.items():
            if i > max_token:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        print("embedding_matrix存储完毕")


        other_info_file = open(other_info_pkl,'wb')
        pickle.dump([[maxlen],[max_token],[embedding_matrix]],other_info_file)
    
        return x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix

'''
Description:判断目标词汇是否为停用词
    args:
        word 带判断的目标词汇
        stopWordList 存储停用词的列表
    return:
        存储停用词的list
'''
def isStopWord(word,stopWordList):
    if word in stopWordList:
        return True
    else:
        return False

'''
Description:获取停用词表
    args:
        filename 存储停用词的文件名
    return:
        存储停用词的list
'''
def getStopWordList(filename):
    stopWordList = [] 
    f = open(filename,encoding='utf-8',errors='ignore')
    for line in f:
        stopWordList.append(line.strip())
    return stopWordList


def main():
    embedding_dims = 300
    path = "./data2/segemnt2word_490w_trainset.txt"
    w2vpath = './w2vmodel1/word2vec(not_all_digit_and_keep_stop_word).model'
    stop_word_filename = './data/stop_word.txt'
    word2idx_pkl = "./data2/word2idx_dict.pkl"
    label2idx_pkl = './data2/label2idx_dict.pkl'
    x_train_dataset_pkl = './data2/x_train_dataset.pkl'
    y_train_dataset_pkl = './data2/y_train_dataset.pkl'
    x_test_dataset_pkl = './data2/x_test_dataset.pkl'
    y_test_dataset_pkl = './data2/y_test_dataset.pkl'
    other_info = './data2/other_info.pkl'
    getdata(path,embedding_dims,w2vpath,stop_word_filename,word2idx_pkl,label2idx_pkl,x_train_dataset_pkl,y_train_dataset_pkl,x_test_dataset_pkl,y_test_dataset_pkl,other_info)
    # pkl_file = open(x_test_dataset_pkl, 'rb')
    # data1 = pickle.load(pkl_file)
    # pprint.pprint(data1)
    # pprint.pprint(len(data1))

if __name__ == '__main__':
    main()