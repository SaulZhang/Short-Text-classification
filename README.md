# Short-Text-classification
第十届大学生服务外包大赛（一等奖解决方案）--A01商品短文本分类。采用基于Keras的Word2vec、CNN、Bi-LSTM、Attention、Adversarial等方法实现商品短文本分类任务。基于Flask框架开发模型的可视化交互软件，支持单条文本以及批量文本的分类处理。

## 1.experiment result
模型在50w数据集上的表现(训练集:测试集=40w:10w)

|  Model   | Accurancy  |
|  ----  | ----  |
| TextCNN  | 0.848 |
| BiLSTM  | 0.860 |
| BiLSTM-Attention (Char Embedding)| 0.838 |
| BiLSTM-Attention (Word Embedding)  | 0.861 |
|  Adversarial-BiLSTM-Attention（Char Embedding）| 0.844|
|  Adversarial-BiLSTM-Attention（Word Embedding）| 0.871 |

模型改进之后的结果（训练集:测试集=40w:10w）
|  Model   | Accurancy  |
|  ----  | ----  |
|  Multi-Head-Attention|   0.9073 |
|   BilSTM   |  0.9156  |
|   0.42BiLSTM+0.58Attention(加权融合)|   0.9194 |
|   0.67BiLSTM+0.09Attention+0.24BiLSTMAttention(加权融合)   |  0.9201  |

## 2.Requirement
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

## 3.dataset & pretrained model
[public training dataset 50w](https://pan.baidu.com/s/1aSy3fxFNvsorfdq2LuK4pA)(提取码：ac2c)<br>
[Attention-wight-norm-WithPositionEmbedding(0.9088).h5](https://pan.baidu.com/s/1vharQoMO2j_6iL0SYcsfLQ)(提取码：tf4a)<br>
[GRUAttention(0.9175799998474121).h5](https://pan.baidu.com/s/1O-VCIsoPzbvol58ngVV43A)(提取码：epnq)<br>
[TextBiLSTM-weightnorm(0.9156999999237061).h5](https://pan.baidu.com/s/1Ub-lcLeAb_EOEqVwStNNVw)(提取码：1u3b)<br>
[word embedding matrix and the sentence length info of dataset](https://pan.baidu.com/s/1QN0e_LsjEvDU2FJ5QeLrow)(提取码：ki3e)<br>

## 4.installation steps of demo
>1、git clone https://github.com/SaulZhang/Short-Text-classification.git <br>
>2、python webGUI.py <br>
>3、在浏览器的地址栏中输入：http://127.0.0.1:8000/

## 5.交互软件使用说明
### 5.1软件名称
商品文本分类(Commodity Text Classfication)

### 5.2软件功能
#### 5.2.1单条分类
在单条数据分类对应的文本输入框内输入商品名称，然后点击“单个数据分类”按钮，等待模型识别，识别结束后将跳转界面，输出分类结果。若要进行下一次分类，请点击“返回”按钮，重复执行上述操作。
#### 5.2.1批量分类
批量分类时，需要选择待识别的文件(该软件仅支持'.txt','.tsv'两种格式的文件，若选择其他格式的文件，软件将给出错误提示)，合法的文件格式为，第一行单独一行为"ITEM_NAME"表示标题(不包含其他分隔符，若文件的内容格式不正确，软件将会给出错误提示，具体内容格式如下图所示)，随后的每一行表示一件商品的名称。待选择正确格式内容的文件之后，点击"批量数据分类"按钮，等待模型识别，识别结束后将跳转界面，输出文件中前200条数据的分类结果。最终识别结果的文件将保存在工程文件夹中的'./result/'文件夹下面。
### 5.3支持浏览器
 Microsoft Edge 41.16299.967.0+、Firefox66.0.1+、Chrome72.0.3626.96+


## 6.Contributor
[@Saul Zhang](https://github.com/SaulZhang)、[@zheng](https://github.com/1029127253)、[@searcher408](https://github.com/Searcher408)、[@nwpuGGBond](https://github.com/nwpu2016303311)、[@Chinazzh8796](https://github.com/Chinazzh8796)
