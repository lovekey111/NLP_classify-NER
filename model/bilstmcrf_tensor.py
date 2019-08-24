import tensorflow as tf
import os
import math
import datetime
from utils import *
from config import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'


# bilstm+crf多任务模型
class Bilstmcrf:
    def __init__(self, batch_size, epoch, lstm_units, embedding_matrix,
                 sequence_len, num_class, train_seq, train_label, test_seq, test_label,
                 train_ner, test_ner, train_tensor, test_tensor, ner_category):
        self.batch_size = batch_size
        self.batch_nums = math.ceil(len(train_seq) / self.batch_size)
        self.epoch = epoch
        self.lstm_units = lstm_units
        self.embedding_matrix = embedding_matrix
        self.sequence_len = sequence_len
        self.num_class = num_class

        self.train_seq = train_seq
        self.train_label = train_label
        self.test_seq = test_seq
        self.test_label = test_label

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

    def add_bilstmcrf_placeholders(self):
        self.e = tf.Variable(self.embedding_matrix, dtype=tf.float32, trainable=False, name='embedding_matrix')
        self.x = tf.placeholder(tf.int32, shape=(None, self.sequence_len), name='input')
        self.y = tf.placeholder(tf.int32, shape=(None, self.num_class), name='label')
        self.y_ner = tf.placeholder(tf.int32, shape=(None, self.sequence_len), name='ner')
        self.word_len = tf.placeholder(tf.int32, shape=(None,), name='word_len')

    def embedding_layer(self):
        with tf.variable_scope('input'):
            embedded = tf.nn.embedding_lookup(self.e, self.x)
        return embedded

    def bilstm_layer(self, lstm_in):
        with tf.variable_scope('bilstm'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, state_is_tuple=True, name='fw_cell')
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, state_is_tuple=True, name='bw_cell')
            (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                       lstm_bw_cell,
                                                                       lstm_in,
                                                                       dtype=tf.float32)
            x_in = tf.concat(outputs, axis=-1)
        return x_in

    def pooling_layer(self, pooling_in):
        with tf.variable_scope('pooling'):
            #常规池化
            # pooling = tf.nn.max_pool1d(x_in, ksize=[1, 2, 2], strides=2, padding='SAME')
            #全局平均池化
            pooling = tf.reduce_max(pooling_in, axis=1)
            out_classify = tf.contrib.layers.fully_connected(pooling, self.num_class, activation_fn=None)
            out_ner = tf.contrib.layers.fully_connected(pooling_in, self.ner_category + 1, activation_fn=None)
        return out_classify, out_ner

    def crf_layer(self, crf_in):
        with tf.variable_scope('crf'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(crf_in,
                                                                                  self.y_ner,
                                                                                  self.word_len)
        return log_likelihood, transition_params

    def bilstm_loss_layer(self, bilstm_loss_in):
        with tf.variable_scope('bilstm_loss'):
            loss = tf.losses.softmax_cross_entropy(logits=bilstm_loss_in,
                                                   onehot_labels=tf.cast(self.y, tf.float32))
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, name='bilstm_optimizer').minimize(loss)
        tf.identity(loss, name='loss')
        return loss, optimizer

    def crf_loss_layer(self, crf_loss_in, crf_likelihood, crf_trans):
        with tf.variable_scope('crf_loss'):
            crf_loss = tf.reduce_mean(-crf_likelihood)
            crf_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(crf_loss)
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(crf_loss_in,
                                                                        crf_trans,
                                                                        self.word_len)
        tf.identity(crf_loss, name='crf_loss_out')
        return crf_loss, crf_optimizer, viterbi_sequence, viterbi_score

    def score_layer(self, bilstm_score_in, crf_score_in):
        with tf.variable_scope('score'):
            # label准确率
            predict_seq, real_seq = tf.argmax(bilstm_score_in, -1), tf.argmax(self.y, -1)
            correct = tf.equal(tf.argmax(bilstm_score_in, -1), tf.argmax(self.y, -1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            # crf准确率
            # crf预测序列
            predict_seq_crf = tf.boolean_mask(crf_score_in, tf.sequence_mask(self.word_len, maxlen=self.sequence_len), name='predict')
            # 真实序列
            real_seq_crf = tf.boolean_mask(self.y_ner, tf.sequence_mask(self.word_len, maxlen=self.sequence_len))
            # bool序列
            crf_acc_bool = tf.equal(predict_seq_crf, real_seq_crf)
        tf.identity(predict_seq, name='predict_seq')
        tf.identity(real_seq, name='real_seq')
        tf.identity(predict_seq_crf, name='predict_seq_crf')
        tf.identity(real_seq_crf, name='real_seq_crf')
        return accuracy, crf_acc_bool

    def build_bilstmcrf_model(self):
        self.add_bilstmcrf_placeholders()
        emb = self.embedding_layer()
        bilstm = self.bilstm_layer(emb)
        classify, ner = self.pooling_layer(bilstm)
        likilhood, trans = self.crf_layer(ner)
        crf_loss, crf_opti, vit_seq, vit_sco = self.crf_loss_layer(ner, likilhood, trans)
        loss, opti = self.bilstm_loss_layer(classify)
        acc, crf_acc_bool = self.score_layer(classify, vit_seq)
        return loss, opti, crf_loss, crf_opti, acc, crf_acc_bool

    def bilstmcrf_train(self, verbose=2):
        classify_loss, classify_optimizer, ner_loss, ner_optimizer, classify_accuracy, ner_accuracy \
            = self.build_bilstmcrf_model()

        x_batch, y_batch, y_ner_batch, train_tensor = \
            self.get_batch_data(self.train_seq, self.train_label, self.train_ner, self.train_tensor)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            lst_acc = []
            test_inp = {self.e: self.embedding_matrix, self.x: self.test_seq, self.y: self.test_label,
                        self.y_ner: self.test_ner, self.word_len: self.test_tensor}
            for e in range(self.epoch):
                all_train_acc, all_train_losses, all_train_acc_crf, all_train_losses_crf = 0, 0, 0, 0
                for b in range(self.batch_nums):
                    data, label, ner, word_size = sess.run([x_batch, y_batch, y_ner_batch, train_tensor])
                    inp = {self.e: self.embedding_matrix, self.x: data, self.y: label,
                           self.y_ner: ner, self.word_len: word_size}
                    sess.run(classify_optimizer, feed_dict=inp)
                    sess.run(ner_optimizer, feed_dict=inp)

                    train_acc = sess.run(classify_accuracy, feed_dict=inp)
                    train_losses = sess.run(classify_loss, feed_dict=inp)
                    train_acc_crf = sess.run(ner_accuracy, feed_dict=inp)
                    train_acc_crf = np.sum(train_acc_crf) / train_acc_crf.shape[0]
                    train_losses_crf = sess.run(ner_loss, feed_dict=inp)

                    all_train_acc += train_acc
                    all_train_losses += train_losses
                    all_train_acc_crf += train_acc_crf
                    all_train_losses_crf += train_losses_crf
                    if verbose == 2:
                        test_acc = sess.run(classify_accuracy, feed_dict=test_inp)
                        test_losses = sess.run(classify_loss, feed_dict=test_inp)
                        print('{} epoch: {}    batch: {}    train_acc: {:.6} train_loss: {:.6} '
                              'test_acc: {:.6} test_loss: {:.6} crf_acc: {:.6} crf_loss: {:.6}'.
                              format(datetime.datetime.now().isoformat(),
                                     e,
                                     b,
                                     train_acc,
                                     train_losses,
                                     test_acc,
                                     test_losses,
                                     train_acc_crf,
                                     train_losses_crf))

                all_train_acc = all_train_acc / self.batch_nums
                all_train_losses = all_train_losses / self.batch_nums
                all_train_acc_crf = all_train_acc_crf / self.batch_nums
                all_train_losses_crf = all_train_losses_crf / self.batch_nums

                test_acc = sess.run(classify_accuracy, feed_dict=test_inp)
                test_losses = sess.run(classify_loss, feed_dict=test_inp)
                print('{} epoch: {}    train_acc: {:.6} train_loss: {:.6} '
                      'test_acc: {:.6} test_loss: {:.6} crf_acc: {:.6} crf_loss: {:.6}'.
                      format(datetime.datetime.now().isoformat(),
                             e,
                             all_train_acc,
                             all_train_losses,
                             test_acc,
                             test_losses,
                             all_train_acc_crf,
                             all_train_losses_crf))
                lst_acc.append(test_acc)
                # 保存模型
                if len(lst_acc) <= 1 or test_acc >= max(lst_acc):
                    saver.save(sess, tensor_model_file)
                    print('model saved')
            coord.request_stop()
            coord.join(threads)
