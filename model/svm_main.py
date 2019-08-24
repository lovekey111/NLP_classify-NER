import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.externals import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from keras_preprocessing.sequence import pad_sequences
from utils import *
from data_process import *
from lstm_model import *
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

WORDSIZE = 300
MAX_SEQUENCE_LEN = 300

# svm文本分类及keras_lstm文本分类主程序
if __name__ == "__main__":
    # 读取词向量
    pre_vec = load_vec(word2vec_file)
    # 数据预处理
    pre_train, pre_train_label, pre_train_index, _ = pre_process(train_file_s, kind='classify', svm=True)
    pre_test, pre_test_label, pre_test_index, _ = pre_process(test_file_s, kind='classify', svm=True)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(pre_train)
    embedding_dic = build_dic(pre_vec, tokenizer.word_index, WORDSIZE,  kind="normal")
    embedding_mat = build_matrix(pre_vec, tokenizer.word_index, WORDSIZE)

    # 一、向量化
    # vec = CountVectorizer(ngram_range=(1, 3), analyzer="word", min_df=5)
    # # vec = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", min_df=5)
    # train_seq = vec.fit_transform(pre_train)
    # test_seq = vec.transform(pre_test)
    # train_seq = train_seq.toarray()
    # test_seq = test_seq.toarray()
    #################################################################################
    # 二、使用嵌入词向量
    train_seq = tokenizer.texts_to_sequences(pre_train)
    test_seq = tokenizer.texts_to_sequences(pre_test)

    # 1、拼接句向量（效果好，但运算时间长）MAX_SEQUENCE_LEN = 300*20
    # train_seq = words_joint(train_seq, embedding_dic)
    # test_seq = words_joint(test_seq, embedding_dic)

    # 2、平均句向量(效果较好)
    # train_seq = words_ave(train_seq, embedding_dic, WORDSIZE)
    # test_seq = words_ave(test_seq, embedding_dic, WORDSIZE)

    # 3、最大句向量（效果较差）
    # train_seq = words_max(train_seq, embedding_dic, WORDSIZE)
    # test_seq = words_max(test_seq, embedding_dic, WORDSIZE)

    # 补齐句向量
    train_seq = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LEN, padding='post', dtype="float")
    test_seq = pad_sequences(test_seq, maxlen=MAX_SEQUENCE_LEN, padding='post', dtype="float")

    # 标签转换
    train_num, test_num = label2num(pre_train_label, pre_test_label)

    # sklearn svm
    # clf = LinearSVC(penalty='l2')
    # clf = SVC(kernel='linear', C=3, probability=True)
    # clf.fit(train_seq, train_num)
    # predict_test = clf.predict(test_seq)
    # predict_prob = clf.predict_proba(test_seq)
    # print(predict_prob)
    # accuracy, precision, recall, fscore = score(predict_test, test_num)
    # print("\n accuray: %f\n precision: %f\n recall: %f\n fscore: %f\n" % (accuracy, precision, recall, fscore))
    # print(classification_report(predict_test, test_num))

    # keras实现lstm
    n_labels = len(set(pre_train_label))
    train_label = to_categorical(train_num, n_labels)
    test_label = to_categorical(test_num, n_labels)

    predict_test = LstmModel(batch_size=64, epoch=10, lstm_units=128, embedding_matrix=embedding_mat,
                             sequence_len=MAX_SEQUENCE_LEN, num_class=n_labels, train_seq=train_seq,
                             train_label=train_label, test_seq=test_seq, test_label=test_label).build_model()
    accuracy, precision, recall, fscore = score(predict_test, test_num)
    print("\n accuray: %f\n precision: %f\n recall: %f\n fscore: %f\n" % (accuracy, precision, recall, fscore))
    print(classification_report(predict_test, test_num))

