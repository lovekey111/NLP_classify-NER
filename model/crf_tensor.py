import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import *
import os
import math
import datetime
import numpy as np
# from bert import modeling
from config import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'


# bilstm+crf命名实体识别模型
class Crf:
    def __init__(self, batch_size, epoch, lstm_units, embedding_matrix,
                 sequence_len, train_seq, test_seq, train_ner, test_ner,
                 train_tensor, test_tensor, ner_category):
        self.batch_size = batch_size
        self.batch_nums = math.ceil(len(train_seq) / self.batch_size)
        self.epoch = epoch
        self.lstm_units = lstm_units
        self.embedding_matrix = embedding_matrix
        self.sequence_len = sequence_len

        self.train_seq = train_seq
        self.test_seq = test_seq

        self.train_ner = train_ner
        self.train_tensor = train_tensor
        self.test_ner = test_ner
        self.test_tensor = test_tensor

        self.ner_category = ner_category

    def get_batch_data(self, *data):
        input_queue = tf.train.slice_input_producer(list(data),
                                                    shuffle=True,
                                                    seed=10)
        all_data = tf.train.batch(input_queue,
                                  batch_size=self.batch_size,
                                  allow_smaller_final_batch=True)
        return all_data

    def add_crf_placeholders(self):
        self.e = tf.Variable(self.embedding_matrix, dtype=tf.float32, trainable=False, name='embedding_matrix')
        self.x = tf.placeholder(tf.int32, shape=(None, self.sequence_len), name='input')
        self.y_ner = tf.placeholder(tf.int32, shape=(None, self.sequence_len), name='ner')
        self.word_len = tf.placeholder(tf.int32, shape=(None,), name='word_len')

    def embedding_layer(self):
        with tf.variable_scope('input'):
            embedded = tf.nn.embedding_lookup(self.e, self.x)
        return embedded

    def bilstm_layer(self, lstm_in, i, batch_norm=False):
        with tf.variable_scope('bilstm_%d' % i):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, state_is_tuple=True, name='fw_cell')
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, state_is_tuple=True, name='bw_cell')
            (outputs_1, output_states_1) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                           lstm_bw_cell,
                                                                           lstm_in,
                                                                           dtype=tf.float32)
            x_in_1 = tf.concat(outputs_1, axis=-1)
        if batch_norm:
            with tf.variable_scope('batch_normalization_%d' % i):
                # self.is_training = tf.placeholder(tf.bool, name='is_training')
                x_in_1 = tf.contrib.layers.layer_norm(x_in_1)
        return x_in_1

    def pooling_layer(self, pooling_in):
        with tf.variable_scope('pooling'):
            out_ner = tf.contrib.layers.fully_connected(pooling_in, self.ner_category + 1, activation_fn=None)
        return out_ner

    def crf_layer(self, crf_in):
        with tf.variable_scope('crf'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(crf_in,
                                                                                  self.y_ner,
                                                                                  self.word_len)
        return log_likelihood, transition_params

    # 增加attention层
    def attention_layer(self, lstm_outs):
        """
        输入lstm的输出组，进行attention处理
        :param lstm_outs:
        :return:
        """
        '''
        w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.seq_len)))
        b_h=tf.Variable(tf.random_normal(shape=(self.seq_len,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        G=tf.nn.softmax(tf.nn.tanh(tf.add(logit,b_h)))#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)
        '''
        w_h = tf.Variable(tf.random_normal(shape=(2*self.lstm_units, 2*self.lstm_units)))
        b_h = tf.Variable(tf.random_normal(shape=(2*self.lstm_units,)))
        logit = tf.einsum("ijk,kl->ijl", lstm_outs, w_h)
        logit = tf.nn.tanh(tf.add(logit, b_h))
        logit = tf.tanh(tf.einsum("ijk,ilk->ijl", logit, lstm_outs))
        G = tf.nn.softmax(logit)  # G.shape=[self.seq_len,self.seq_len]
        logit_ = tf.einsum("ijk,ikl->ijl", G, lstm_outs)
        # 注意力得到的logit与lstm_outs进行链接
        outs = tf.concat((logit_, lstm_outs), 2)  # outs.shape=[None,seq_len,4*hidden_dim]
        return outs

    def crf_loss_layer(self, crf_loss_in, crf_likelihood, crf_trans, batch_norm=False):
        with tf.variable_scope('loss'):
            crf_loss = tf.reduce_mean(-crf_likelihood)
            if batch_norm:
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    crf_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(crf_loss)
            else:
                crf_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(crf_loss)
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(crf_loss_in,
                                                                        crf_trans,
                                                                        self.word_len)
        tf.identity(crf_loss, name='crf_loss')
        return crf_loss, crf_optimizer, viterbi_sequence, viterbi_score

    def score_layer(self, crf_score_in):
        with tf.variable_scope('score'):
            # crf准确率
            # crf预测序列
            predict_seq = tf.boolean_mask(crf_score_in, tf.sequence_mask(self.word_len, maxlen=self.sequence_len), name='predict')
            # 真实序列
            real_seq = tf.boolean_mask(self.y_ner, tf.sequence_mask(self.word_len, maxlen=self.sequence_len))
            # bool序列
            crf_acc_bool = tf.equal(predict_seq, real_seq)

        tf.identity(predict_seq, name='predict_seq')
        tf.identity(real_seq, name='real_seq')
        return crf_acc_bool

    def build_crf_model(self):
        self.add_crf_placeholders()
        emb = self.embedding_layer()
        bilstm = self.bilstm_layer(emb, i=1)
        ner = self.pooling_layer(bilstm)
        likilhood, trans = self.crf_layer(ner)
        crf_loss, crf_opti, vit_seq, vit_sco = self.crf_loss_layer(ner, likilhood, trans)
        crf_acc_bool = self.score_layer(vit_seq)
        return crf_loss, crf_opti, crf_acc_bool

    def build_crf_attention_model(self):
        self.add_crf_placeholders()
        emb = self.embedding_layer()
        bilstm = self.bilstm_layer(emb, i=1)
        # 注意力
        att = self.attention_layer(bilstm)
        # 多头注意力
        # att = modeling.attention_layer(bilstm, bilstm, num_attention_heads=3)
        ner = self.pooling_layer(att)
        likilhood, trans = self.crf_layer(ner)
        crf_loss, crf_opti, vit_seq, vit_sco = self.crf_loss_layer(ner, likilhood, trans)
        crf_acc_bool = self.score_layer(vit_seq)
        return crf_loss, crf_opti, crf_acc_bool

    def build_crf_layer_norm_model(self):
        self.add_crf_placeholders()
        emb = self.embedding_layer()
        bilstm = self.bilstm_layer(emb, i=1, batch_norm=True)
        ner = self.pooling_layer(bilstm)
        likilhood, trans = self.crf_layer(ner)
        crf_loss, crf_opti, vit_seq, vit_sco = self.crf_loss_layer(ner, likilhood, trans, batch_norm=True)
        crf_acc_bool = self.score_layer(vit_seq)
        return crf_loss, crf_opti, crf_acc_bool

    def crf_train(self, verbose=2):
        # ner_loss, ner_optimizer, ner_accuracy = self.build_crf_model()
        ner_loss, ner_optimizer, ner_accuracy = self.build_crf_attention_model()
        # ner_loss, ner_optimizer, ner_accuracy = self.build_crf_layer_norm_model()
        x_batch, y_ner_batch, train_tensor = self.get_batch_data(self.train_seq, self.train_ner, self.train_tensor)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            lst_acc = []

            test_inp = {self.e: self.embedding_matrix, self.x: self.test_seq,
                        self.y_ner: self.test_ner, self.word_len: self.test_tensor}

            for e in range(self.epoch):
                all_train_acc_crf, all_train_losses_crf = 0, 0
                for b in range(self.batch_nums):
                    data, ner, word_size = sess.run([x_batch, y_ner_batch, train_tensor])
                    inp = {self.e: self.embedding_matrix, self.x: data,
                           self.y_ner: ner, self.word_len: word_size}
                    sess.run(ner_optimizer, feed_dict=inp)

                    train_acc_crf = sess.run(ner_accuracy, feed_dict=inp)
                    train_acc_crf = np.sum(train_acc_crf) / train_acc_crf.shape[0]
                    train_losses_crf = sess.run(ner_loss, feed_dict=inp)

                    all_train_acc_crf += train_acc_crf
                    all_train_losses_crf += train_losses_crf

                    if verbose == 2:
                        test_acc_crf = sess.run(ner_accuracy, feed_dict=test_inp)
                        test_acc_crf = np.sum(test_acc_crf) / test_acc_crf.shape[0]
                        test_losses_crf = sess.run(ner_loss, feed_dict=test_inp)
                        print('{} epoch: {}    batch: {}    train_acc: {:.6} train_loss: {:.6} test_acc: {:.6} '
                              'test_loss: {:.6}'.format(datetime.datetime.now().isoformat(),
                                                        e,
                                                        b,
                                                        train_acc_crf,
                                                        train_losses_crf,
                                                        test_acc_crf,
                                                        test_losses_crf))

                all_train_acc_crf = all_train_acc_crf / self.batch_nums
                all_train_losses_crf = all_train_losses_crf / self.batch_nums

                test_acc_crf = sess.run(ner_accuracy, feed_dict=test_inp)
                test_acc_crf = np.sum(test_acc_crf) / test_acc_crf.shape[0]
                test_losses_crf = sess.run(ner_loss, feed_dict=test_inp)
                print('{} epoch: {}    train_acc: {:.6} train_loss: {:.6} test_acc: {:.6} test_loss: {:.6} '.
                      format(datetime.datetime.now().isoformat(),
                             e,
                             all_train_acc_crf,
                             all_train_losses_crf,
                             test_acc_crf,
                             test_losses_crf))
                lst_acc.append(test_acc_crf)
                # 保存模型
                if len(lst_acc) <= 1 or test_acc_crf >= max(lst_acc):
                    saver.save(sess, tensor_model_file)
                    print('model saved')
            coord.request_stop()
            coord.join(threads)
