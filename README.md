# Short-Text-classification
第十届大学生服务外包大赛（一等奖解决方案）--A01商品短文本分类。基于Word2vec、CNN、Bi-LSTM、Attention、Adversarial等方法实现商品短文本分类任务。

## experiment result
模型在50w数据集上的表现(训练集:测试集=40w:10w)

|  Model   | Accurancy  |
|  ----  | ----  |
| TextCNN  | 0.848 |
| BiLSTM  | 0.860 |
| BiLSTM-Attention (Char Embedding)| 0.838 |
| BiLSTM-Attention (Word Embedding)  | 0.861 |
|  Adversarial-BiLSTM-Attention（Char Embedding）| 0.844|
|  Adversarial-BiLSTM-Attention（Word Embedding）| 0.871 |

模型改进之后的结果（训练集:40w+半监督+爬虫数据验证集:10w）
|  Model   | Accurancy  |
|  ----  | ----  |
|  Multi-Head-Attention|   0.9073 |
|   BilSTM   |  0.9156  |
|   0.42BiLSTM+0.58Attention(加权融合)|   0.9194 |
|   0.67BiLSTM+0.09Attention+0.24BiLSTMAttention(加权融合)   |  0.9201  |

## Requirement
> Keras==2.0.5+
Python3.6+
>pandas==0.20.3
Flask==0.12.2
xlrd==1.1.0
jieba==0.39
tensorflow==1.4.0
h5py==2.7.0
Keras==2.0.5
numpy==1.14.2

## Contributor
[@Saul Zhang](https://github.com/SaulZhang)、[@zheng](https://github.com/1029127253)、[@searcher408](https://github.com/Searcher408)、[@nwpuGGBond](https://github.com/nwpu2016303311)
