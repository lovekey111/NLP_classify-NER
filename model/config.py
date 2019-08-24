PATH = ".."
# 数据格式（用于分类）
# 时间：数字：数字：数字：数字：类别：文本句子
# 20181207154915:232961161:843099:843099:843099:901590:已上传
# 小样本数据：train-1173行、test-131行
train_file_s = PATH+"/data/small/rg_c2_train_20181204_1000002_1000002843099.train"
test_file_s = PATH+"/data/small/rg_c2_train_20181204_1000002_1000002843099.test"
# 测试数据：20行
test_similar_file = PATH+"/data/small/similar.test"
# 大样本数据：train-325247行、test-26399行
train_file_b = PATH+"/data/big/rg_train_20190121_1000002.train"
test_file_b = PATH+"/data/big/rg_train_20190121_1000002.test"
# 大样本数据：train-325247行、test-26399行
train_file_n = PATH+"/data/new/rg_train_20190701_1000002.train"
test_file_n = PATH+"/data/new/rg_train_20190701_1000002.test"
###################################################################################
# 数据格式（用于分类和实体识别）
# 时间：数字：数字：数字：数字：类别：词1 词2 词3...：实体标注1 实体标注2 实体标注3...
# 20190116170655:4425831:1000002:2000001:3000001:4000001:我 提现 失败 了:O XW O O
# 样本数据：train-250496行、test-18034行
train_file_ner = PATH+"/data/ner/rg_train_20190701_1000002.train_ner"
test_file_ner = PATH+"/data/ner/rg_train_20190701_1000002.test_ner"
###################################################################################
# 数据格式（用于实体识别）
# 时间：数字：数字：数字：数字：类别：文本句子：实体标注1 实体标注2 实体标注3...：标注词1&标注词2&标注词3
# 20190311192721:4783526:1000002:2000010:3000024:4000142:我不想再续费怎么操作:O B E S B E O O O O:不想&再&续费
# 样本数据：train-654行、test-27行
zdxf_train_ner_file = PATH+"/data/new_ner/cp_4000142_0730_自动续费.train.txt"
zdxf_test_ner_file = PATH+"/data/new_ner/cp_4000142_0730_自动续费.test.txt"
# 样本数据：train-1726行、test-192行
new_train_ner_file = PATH+"/data/new_ner/train.txt"
new_test_ner_file = PATH+"/data/new_ner/test.txt"
###################################################################################
# 时间：数字：数字：数字：数字：类别：文本句子：实体标注1 实体标注2 实体标注3...：标注词1&标注词2&标注词3
# ::::::商家没有退款，所以以后再不相信你们的平台:B E S O B E O O O O O O O O O O O O O O:商家&没&退款
# 样本数据：10013行
ner_origin_file = PATH+"/data/new_ner/divide/train_origin.txt"
# 样本数据：train-9011行、test-1002行
ner_91_train_file = PATH+"/data/new_ner/divide/train_91.txt"
ner_91_test_file = PATH+"/data/new_ner/divide/test_91.txt"
# 样本数据：train-7009行、test-3004行
ner_73_train_file = PATH+"/data/new_ner/divide/train_73.txt"
ner_73_test_file = PATH+"/data/new_ner/divide/test_73.txt"
# 样本数据：train-5007行、test-5006行
ner_55_train_file = PATH+"/data/new_ner/divide/train_55.txt"
ner_55_test_file = PATH+"/data/new_ner/divide/test_55.txt"
# 样本数据：train-3003行、test-7010行
ner_37_train_file = PATH+"/data/new_ner/divide/train_37.txt"
ner_37_test_file = PATH+"/data/new_ner/divide/test_37.txt"
# 样本数据：train-1001行、test-9012行
ner_19_train_file = PATH+"/data/new_ner/divide/train_19.txt"
ner_19_test_file = PATH+"/data/new_ner/divide/test_19.txt"
###################################################################################
# 停用词文件
stopwords_file = PATH+"/data/stopwords.txt"

word2vec_model_file = PATH+"/data/model/train_big.model"
word2vec_model_txt_file = PATH+"/data/model/train_big.model.txt"
word2vec_vocab_txt_file = PATH+"/data/model/train_big.vocab.txt"
# 词向量文件
word2vec_file = PATH+"/data/w2v_ai_session_5month_output_wiki_20.pkl"

# 模型文件
svm_model_file = PATH+"/data/result_model/svm_model.pkl"
lstm_model_file = PATH+"/data/result_model/lstm_model.h5"

# tensorflow模型
rate = '91'
tensor_model_file = PATH+"/data/result_model/tensor_model/tensor_model" + rate + ".ckpt"
checkpoint_file = PATH+"/data/result_model/tensor_model/checkpoint"
tensor_data_file = PATH+"/data/result_model/tensor_model/tensor_model" + rate + ".ckpt.data-00000-of-00001"
tensor_index_file = PATH+"/data/result_model/tensor_model/tensor_model" + rate + ".ckpt.index"
tensor_meta_file = PATH+"/data/result_model/tensor_model/tensor_model" + rate + ".ckpt.meta"

# 预测结果文件
lstm_result_file = PATH+"/data/result/lstm_result.csv"
svm_result_file = PATH+"/data/result/svm_result.csv"

