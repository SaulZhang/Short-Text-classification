# -*- coding: utf-8 -*-  
'''
实现数据的预处理
'''
import jieba
import pickle,pprint
import re
import string
import sys

'''
description:对原文本进行分词处理。返回存储分词以及出现次数的字典
	args:
		filename 存放训练数据的文件名
		stop_word_filename 存放停用词的文件名
		segemntfilename 存储分割后分词的文件
	returns:
		按照出现频率对分词进行排序的分词词典
'''
def fenci(filename1,filename2,stop_word_filename,segemntfilename):
	dict_data = {}
	stopWordList = []
	f = open(filename1,encoding='utf-8')
	# stopWordList = getStopWordList(stop_word_filename)

	output = open(segemntfilename, 'w', encoding='utf-8')

	for k,line in enumerate(f):
		# if k > 1000:break
		sample,target = line.split("\t")
		sample = sample.strip()
		#除去干扰符号以及标点符号
		# sample = re.sub("★[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+".encode('utf-8').decode("utf-8"), "",sample.encode('utf-8').decode("utf-8"))
		sample = sample.translate(str.maketrans('', '',"★[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+" ))
		seg_list = jieba.cut(sample,cut_all=False)
		seg_list = list(seg_list)
		print("process line:",k)
		#将分词存储在字典中
		for element in seg_list:
			# if isStopWord(element,stopWordList) == True:
			# 	continue
			if (element).isdigit()==True:
				continue
			output.write(element+' ')  #按照word2vec所需的格式对数据进行存储，格式为:每个句子进行分词处理，词与词之间用空格进行分隔
		output.write('\t'+target+'\n')

	output.close()
	#对字典中的分词按照其出现的次数进行降序排序
	sorted_dict=sorted(dict_data.items(),key = lambda x:x[1],reverse = True)
	
	return sorted_dict


'''
description:对原文本进行进行标签提取处理
	args:
		filename 存放训练数据的文件名
	returns:
		按照出现频率对分词进行排序的标签词典
'''
def getLabel(filename):
	dict_label = {}
	f = open(filename,encoding='gb18030',errors='ignore')
	for k,line in enumerate(f):
		if k==0:continue
		# if k > 1000:break
		sample,target = line.strip().split('\t')
		linelist = target.split("--")	#分割多标签
		#将标签存储在字典中
		for e in linelist:
			if e not in dict_label:
				dict_label[e] = 0
			else:
				dict_label[e] += 1

	sorted_dict=sorted(dict_label.items(),key = lambda x:x[1],reverse = True)	#对字典中的标签按照其出现的次数进行降序排序
	return sorted_dict


'''
description:将训练样本中的每一句话转化为对应的分词序号+标签序号的形式，分词与分词之间用一个空格分割，
			分词与标签之间用一个tab键分割，标签与标签之间用-进行分割
	args:
		traindatafilename 存放训练数据的文件名
		savefilename 存放最后结果的文件名
		stop_word_filename 存放停用词的文件名
		fenci_idx 分词及其序号的字典(k,v)对
		label_idx 标签及其序号的字典(k,v)对

'''
def sentence2idx(traindatafilename,savefilename,segemntfilename,fenci_idx,label_idx):
	dict_data = {}
	stopWordList = []
	try:
		f = open(traindatafilename,encoding='gb18030',errors='ignore')
		f_result = open(savefilename,'w')
	except:
		print("file open exception!")
		sys.exit(-1)

	stopWordList = getStopWordList(stop_word_filename)

	for k,line in enumerate(f):
		if k==0:continue
		# if k > 1000:break
		sample,target = line.strip().split("\t")
		#除去干扰符号以及标点符号
		# sample = re.sub("★[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+".encode('utf-8').decode("utf-8"), "",sample.encode('utf-8').decode("utf-8"))
		sample = sample.translate(str.maketrans('', '',"★[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+" ))
		seg_list = jieba.cut(sample,cut_all=False)
		seg_list = list(seg_list)
		line = ""
		for element in seg_list:
			#判断是否为停用词
			if isStopWord(element,stopWordList) == True:
				continue
			if element in fenci_idx.keys():
				line += fenci_idx[element]
			else:
				line += "-1"
			line += " "
		line = line[0:-1] #去除行尾部的空格
		line = line + "\t"
		linelist = target.split("--")	#分割多标签
		#将标签存储在字典中
		for e in linelist:
			if e not in label_idx.keys():
				line += "-1"+"--"
			else:
				line += label_idx[e]+"--"
		line = line[0:-2]
		f_result.write(line+"\n")
	f_result.close()


'''
Description:获取停用词表
	args:
		filename 存储停用词的文件名
	return:
		存储停用词的list
'''
def getStopWordList(filename):
	stopWordList = [] 
	f = open(filename,encoding='gb18030',errors='ignore')
	for line in f:
		stopWordList.append(line.strip())
	return stopWordList


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



if __name__ == '__main__':

	train_data_filename1 = "./data2/test-10w.tsv"
	train_data_filename2 = "./data2/train_set_450w.txt"
	stop_word_filename = ""
	segemnt_word_filename = "./data2/segemnt2word_10w.txt"
	jieba.set_dictionary('./data/dict.txt.big')
	fenci_count_dict = fenci(train_data_filename1,train_data_filename2,stop_word_filename,segemnt_word_filename)