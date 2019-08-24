import tensorflow as tf
import os
import math
import datetime
from config import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'


# bilstm文本分类模型
class Bilstm:
    def __init__(self, batch_size, epoch, lstm_units, embedding_matrix, sequence_len, num_class,
                 train_seq, train_label, test_seq, test_label):
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

    def get_batch_data(self, *data):
        input_queue = tf.train.slice_input_producer(list(data),
                                                    shuffle=True,
                                                    seed=10)
        all_data = tf.train.batch(input_queue,
                                  batch_size=self.batch_size,
                                  allow_smaller_final_batch=False)
        return all_data

    def add_bilstm_placeholders(self):
        self.e = tf.Variable(self.embedding_matrix, dtype=tf.float32, trainable=False, name='embedding_matrix')
        self.x = tf.placeholder(tf.int32, shape=(None, self.sequence_len), name='input')
        self.y = tf.placeholder(tf.int32, shape=(None, self.num_class), name='label')

    def embedding_layer(self):
        with tf.variable_scope('input'):
            embedded = tf.nn.embedding_lookup(self.e, self.x)
        return embedded

    def bilstm_layer(self, lstm_in, batch_norm=False):
        with tf.variable_scope('bilstm'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, state_is_tuple=True, name='fw_cell')
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, state_is_tuple=True, name='bw_cell')
            (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                       lstm_bw_cell,
                                                                       lstm_in,
                                                                       dtype=tf.float32)
            x_in = tf.concat(outputs, axis=-1)
        if batch_norm:
            with tf.variable_scope('batch_normalization'):
                self.is_training = tf.placeholder(tf.bool, name='is_training')
                batch_norm = tf.layers.batch_normalization(x_in, training=self.is_training)
            return batch_norm
        return x_in

    def pooling_layer(self, pooling_in):
        with tf.variable_scope('pooling'):
            #常规池化
            # pooling = tf.nn.max_pool1d(x_in, ksize=[1, 2, 2], strides=2, padding='SAME')
            #全局平均池化
            pooling = tf.reduce_max(pooling_in, axis=1)
            out = tf.contrib.layers.fully_connected(pooling, self.num_class, activation_fn=None)
        return out

    def bilstm_loss_layer(self, loss_in, batch_norm=False):
        with tf.variable_scope('bilstm_loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=loss_in, labels=self.y), name='loss')
            if batch_norm:
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, name='optimizer').minimize(loss)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, name='optimizer').minimize(loss)
        tf.identity(loss, name='loss')
        return loss, optimizer

    def bilstm_score_layer(self, score_in):
        with tf.variable_scope('bilstm_score'):
            predict_seq, real_seq = tf.argmax(score_in, -1), tf.argmax(self.y, -1)
            # 准确率
            correct = tf.equal(tf.argmax(score_in, -1), tf.argmax(self.y, -1), name='corr')
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')
        tf.identity(predict_seq, name='predict_seq')
        tf.identity(real_seq, name='real_seq')
        return accuracy

    def build_bilstm_model(self):
        self.add_bilstm_placeholders()
        emb = self.embedding_layer()
        bilstm = self.bilstm_layer(emb)
        pool = self.pooling_layer(bilstm)
        loss, opti = self.bilstm_loss_layer(pool)
        acc = self.bilstm_score_layer(pool)
        return loss, opti, acc

    def bilstm_train(self, verbose=2):
        loss, optimizer, accuracy = self.build_bilstm_model()
        x_batch, y_batch = self.get_batch_data(self.train_seq, self.train_label)
        saver = tf.train.Saver()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            test_inp = {self.e: self.embedding_matrix, self.x: self.test_seq, self.y: self.test_label}
            lst_acc = []
            for e in range(self.epoch):
                all_train_acc, all_train_losses = 0, 0
                for b in range(self.batch_nums):
                    data, label = sess.run([x_batch, y_batch])
                    inp = {self.e: self.embedding_matrix, self.x: data, self.y: label}
                    sess.run(optimizer, feed_dict=inp)

                    train_acc = sess.run(accuracy, feed_dict=inp)
                    train_losses = sess.run(loss, feed_dict=inp)
                    all_train_acc += train_acc
                    all_train_losses += train_losses

                    if verbose == 2:
                        test_acc = sess.run(accuracy, feed_dict=test_inp)
                        test_losses = sess.run(loss, feed_dict=test_inp)
                        print('{} epoch: {}    batch: {}    train_acc: {:.6} train_loss: {:.6} test_acc: {:.6} '
                              'test_loss: {:.6}'.format(datetime.datetime.now().isoformat(),
                                                        e,
                                                        b,
                                                        train_acc,
                                                        train_losses,
                                                        test_acc,
                                                        test_losses))
                all_train_acc = all_train_acc / self.batch_nums
                all_train_losses = all_train_losses / self.batch_nums
                test_acc = sess.run(accuracy, feed_dict=test_inp)
                test_losses = sess.run(loss, feed_dict=test_inp)
                print('{} epoch: {}    train_acc: {:.6} train_loss: {:.6} test_acc: {:.6} test_loss: {:.6}'.
                      format(datetime.datetime.now().isoformat(),
                             e,
                             all_train_acc,
                             all_train_losses,
                             test_acc,
                             test_losses))
                lst_acc.append(test_acc)
                # 保存模型
                if len(lst_acc) <= 1 or test_acc >= max(lst_acc):
                    saver.save(sess, tensor_model_file)
                    print('model saved')
            coord.request_stop()
            coord.join(threads)
