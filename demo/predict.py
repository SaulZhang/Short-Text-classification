'''
Date:2019-2-7
Author:SaulZhang
Description:Web可视化界面与模型交互的接口,用于预测单条数据和批量的数据
'''
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import jieba
import pickle,pprint
import string

#将单个句子转变为word2idx的形式
def processSingleExample(string,word2idx,maxlen):
	sample = string.strip()
	sample = sample.translate(str.maketrans('', '',"★[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+" ))
	sample = sample.strip('\n')
	seg_list = jieba.cut(sample,cut_all=False)
	seg_list = list(seg_list)
	new_seg_list = []
	for element in seg_list:
		if (element).isdigit()==True:
			continue
		else:
			new_seg_list.append(element)

	word2idx_pkl = open(word2idx, 'rb')
	word2idxdict = pickle.load(word2idx_pkl)
	idxlist = []
	print(new_seg_list)
	for word in new_seg_list:
		try:
			idxlist.append(word2idxdict[word])
		except:
			idxlist.append(0)#该分词在字典中不存在
	print(idxlist)
	idxlist = pad_sequences([idxlist],maxlen=maxlen)
	print(idxlist)
	return idxlist


#将多个句子转变为word2idx的形式
def processMultiExample(testfilename,word2idxfilename,maxlen):
    print('Loading multi-data...')
    countline = 0
    try:
        d = pd.read_csv(testfilename,encoding='gb18030',sep = '\t')
        d.columns=['title']
    #drop=True 不生成index列
        d=d[-pd.isnull(d["title"])].reset_index(drop=True)
        dx=[]
        word_to_id_dataset = open(word2idxfilename, 'rb')
        word_to_id = pickle.load(word_to_id_dataset)
        for idx,line in enumerate(d["title"]):
            line = line.translate(str.maketrans('', '',"★[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+" ))
            line = line.strip('\n')
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
    except:
        return "FILE_FORMAT_ERROR"
    # print("x_test:",x_test)
    

def main():
    pass

if __name__ == '__main__':
    main()