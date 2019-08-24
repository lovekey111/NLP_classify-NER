from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.externals import joblib
from sklearn.svm import SVC, LinearSVC
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.metrics import classification_report
from utils import *
from data_process import *
from lstm_model import *
from config import *
from bilstm_tensor import Bilstm
from bilstmcrf_tensor import Bilstmcrf
from crf_tensor import Crf
import logging
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
warnings.filterwarnings("ignore")

WORDSIZE = 300
MAX_SEQUENCE_LEN = 20
dic_ner = {'FJ': 1, 'GN': 2, 'O': 3, 'SP': 4, 'XW': 5, 'YW': 6}
dic_sbieo = {'S': 1, 'B': 2, 'I': 3, 'E': 4, 'O': 5}

# tensorflow模型主程序

# 输入， 标签， ner标签， 有效长度
def test_model(*params, model='crf_tensor'):
    saver = tf.train.import_meta_graph(tensor_meta_file)
    graph = tf.get_default_graph()

    if model == 'crf_tensor':
        if len(params) != 3:
            raise ValueError('Number of params is wrong')
        else:
            x_t, y_ner_t, word_len_t = params[0], params[1], params[2]
        # 输入
        x = graph.get_tensor_by_name('input:0')
        y_ner = graph.get_tensor_by_name('ner:0')
        word_len = graph.get_tensor_by_name('word_len:0')
        # 输出
        real_seq = graph.get_tensor_by_name('real_seq:0')
        test_loss = graph.get_tensor_by_name('crf_loss:0')
        predict_seq = graph.get_tensor_by_name('predict_seq:0')

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('../data/result_model/tensor_model'))
            inp = {x: x_t, y_ner: y_ner_t, word_len: word_len_t}

            pre_seq = sess.run(predict_seq, feed_dict=inp)
            real_seq = sess.run(real_seq, feed_dict=inp)
            loss = sess.run(test_loss, feed_dict=inp)
    elif model == 'bilstm_tensor':
        if len(params) != 2:
            raise ValueError('Number of params is wrong')
        else:
            x_t, y_t = params[0], params[1]
        # 输入
        x = graph.get_tensor_by_name('input:0')
        y = graph.get_tensor_by_name('label:0')
        # 输出
        test_loss = graph.get_tensor_by_name('loss:0')
        predict_seq = graph.get_tensor_by_name('predict_seq:0')
        real_seq = graph.get_tensor_by_name('real_seq:0')

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('../data/result_model/tensor_model'))
            inp = {x: x_t, y: y_t}

            pre_seq = sess.run(predict_seq, feed_dict=inp)
            real_seq = sess.run(real_seq, feed_dict=inp)
            loss = sess.run(test_loss, feed_dict=inp)
    elif model == 'bilstmcrf_tensor':
        if len(params) != 4:
            raise ValueError('Number of params is wrong')
        else:
            x_t, y_t, y_ner_t, word_len_t = params[0], params[1], params[2], params[3]
        # 输入
        x = graph.get_tensor_by_name('input:0')
        y = graph.get_tensor_by_name('label:0')
        y_ner = graph.get_tensor_by_name('ner:0')
        word_len = graph.get_tensor_by_name('word_len:0')
        # 输出
        test_loss = graph.get_tensor_by_name('loss:0')
        test_loss_crf = graph.get_tensor_by_name('crf_loss_out:0')
        predict_seq = graph.get_tensor_by_name('predict_seq:0')
        real_seq = graph.get_tensor_by_name('real_seq:0')

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('../data/result_model/tensor_model'))
            inp = {x: x_t, y: y_t, y_ner: y_ner_t, word_len: word_len_t}

            pre_seq = sess.run(predict_seq, feed_dict=inp)
            real_seq = sess.run(real_seq, feed_dict=inp)
            loss = sess.run(test_loss, feed_dict=inp)
            crf_loss = sess.run(test_loss_crf, feed_dict=inp)

    evaluate(real_seq, pre_seq, kind='test_ner')
    print(classification_report(real_seq, pre_seq))
    print('loss: %f' % loss)


if __name__ == "__main__":
    # 加载预训练词向量
    pre_vec = load_vec(word2vec_file)

    # #####################读取数据####################
    pre_train, pre_train_label, pre_train_index, pre_train_ner = pre_process(train_file_ner, kind='mul_ner')
    pre_test, pre_test_label, pre_test_index, pre_test_ner = pre_process(test_file_ner, kind='mul_ner')
    ##################################################
    # 去除低频词
    # pre_train = count_word(pre_train, 5)
    # pre_test = count_word(pre_test)
    logging.info("Loaded files")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(pre_train)
    # 构建嵌入词典
    # embedding_dic = build_dic(pre_vec, tokenizer.word_index, kind="normal")
    # 构建嵌入矩阵
    embedding_mat = build_matrix(pre_vec, tokenizer.word_index, WORDSIZE)

    train_seq = tokenizer.texts_to_sequences(pre_train)
    test_seq = tokenizer.texts_to_sequences(pre_test)
    train_seq = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LEN, padding='post', dtype="float")
    test_seq = pad_sequences(test_seq, maxlen=MAX_SEQUENCE_LEN, padding='post', dtype="float")

    if pre_train_label and pre_test_label:
        n_labels = len(set(pre_train_label))

        train_num, test_num = label2num(pre_train_label, pre_test_label)
        train_label = to_categorical(train_num, n_labels)
        test_label = to_categorical(test_num, n_labels)
    else:
        print('Don\'t have label, NER')

    if pre_train_ner and pre_test_ner:
        n_tags = len(list(dic_ner.keys()))

        pre_train_ner, pre_test_ner = ner2num(pre_train_ner, dic_ner), ner2num(pre_test_ner, dic_ner)
        train_ner_tmp = pad_sequences(maxlen=MAX_SEQUENCE_LEN, sequences=pre_train_ner, padding='post')
        train_ner = [to_categorical(label, num_classes=n_tags + 1) for label in train_ner_tmp]
        test_ner_tmp = pad_sequences(maxlen=MAX_SEQUENCE_LEN, sequences=pre_test_ner, padding='post')
        test_ner = [to_categorical(label, num_classes=n_tags + 1) for label in test_ner_tmp]

        train_ner_tensor = list(map(lambda x: len(x) if len(x) <= MAX_SEQUENCE_LEN else MAX_SEQUENCE_LEN, pre_train_ner))
        test_ner_tensor = list(map(lambda x: len(x) if len(x) <= MAX_SEQUENCE_LEN else MAX_SEQUENCE_LEN, pre_test_ner))
    else:
        print('Don\'t have ner, classify')


    # Bilstm
    # Bilstm(batch_size=64, epoch=10, lstm_units=128, embedding_matrix=embedding_mat,
    #        sequence_len=MAX_SEQUENCE_LEN, num_class=n_labels, train_seq=train_seq,
    #        train_label=train_label, test_seq=test_seq, test_label=test_label).bilstm_train()
    # 模型测试
    # test_model(test_seq, test_label, model='bilstm_tensor')

    # Bilstmcrf多任务
    Bilstmcrf(batch_size=256, epoch=1, lstm_units=128, embedding_matrix=embedding_mat,
              sequence_len=MAX_SEQUENCE_LEN, num_class=n_labels, train_seq=train_seq,
              train_label=train_label, test_seq=test_seq, test_label=test_label, train_ner=train_ner_tmp,
              test_ner=test_ner_tmp, train_tensor=train_ner_tensor, test_tensor=test_ner_tensor, ner_category=n_tags)\
        .bilstmcrf_train()
    # 模型测试
    test_model(test_seq, test_label, test_ner_tmp, test_ner_tensor, model='bilstmcrf_tensor')

    # CRF
    # Crf(batch_size=64, epoch=25, lstm_units=128, embedding_matrix=embedding_mat, sequence_len=MAX_SEQUENCE_LEN,
    #     train_seq=train_seq, test_seq=test_seq, train_ner=train_ner_tmp, test_ner=test_ner_tmp,
    #     train_tensor=train_ner_tensor, test_tensor=test_ner_tensor, ner_category=n_tags)\
    #     .crf_train()
    # # 模型测试
    # test_model(test_seq, test_ner_tmp, test_ner_tensor)

