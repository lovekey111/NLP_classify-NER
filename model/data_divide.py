from model.data_process import *
from model.config import *
# 按比例划分数据集
divide_data(ner_origin_file, ner_91_train_file, ner_91_test_file, train_rate=0.9)