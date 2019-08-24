# NLP_classify-NER
使用keras与tensorflow搭建中文文本分类与命名实体识别模型

## 1、需安装包
* 必备
	*  numpy:  矩阵及数据处理
	* jieba: 中文分词
* SVM必备
	* sklearn: 机器学习
* 神经网络必备
	* tensorflow: google深度学习框架
	* keras及keras_contrib: 基于tensorflow封装的深度学习高级API
* 可选
	* gensim: 词向量训练
	
## 2、文件说明
* data: 存放训练数据、预训练词向量以及模型文件。(需自行准备数据，数据格式在model/config.py中有说明)
* model: 数据预处理及模型训练
	* bilstm_tensor.py: 基于tensorflow的双向lstm文本分类模型
	* bilstmcrf_tensor.py: 基于tensorflow的双向lstm+crf(条件随机场)文本分类及命名实体识别多任务模型
	* config.py: 文件及模型存放路径
	* crf_tensor.py: 基于tensorflow的双向lstm+crf(条件随机场)命名实体识别模型
	* data_divide.py: 按比例划分数据集
	* data_process.py: 数据预处理
	* lstm_model.py: 基于kears搭建文本分类、命名实体识别及多任务模型
	* svm_main.py: 主程序，训练svm文本分类及lstm_model.py中模型
	* tensor_main.py: 主程序，训练bilstm_tensor.py、bilstmcrf_tensor.py、crf_tensor.py中模型
	* utils.py: 模型评估
