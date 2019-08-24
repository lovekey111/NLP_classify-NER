import os
from config import *
import numpy as np
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras_contrib.layers import CRF
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, LSTM, GlobalMaxPool1D, Bidirectional, CuDNNLSTM, \
    SpatialDropout1D, Flatten, Input, TimeDistributed, Dropout, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# keras实现的三种模型
class LstmModel:
    def __init__(self, batch_size, epoch, lstm_units, embedding_matrix, sequence_len, num_class,
                 train_seq, train_label, test_seq, test_label):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lstm_units = lstm_units
        self.embedding_matrix = embedding_matrix
        self.sequence_len = sequence_len
        self.num_class = num_class

        self.train_seq = train_seq
        self.train_label = train_label
        self.test_seq = test_seq
        self.test_label = test_label

        self.checkpoint = ModelCheckpoint(filepath=lstm_model_file, monitor='dense_acc', verbose=1, save_best_only=True)

    #双向lstm模型
    def build_model(self):
        checkpoint = ModelCheckpoint(filepath=lstm_model_file, monitor='acc', verbose=1, save_best_only=True)
        words = Input(shape=(None, ))
        x = Embedding(*self.embedding_matrix.shape,
                      weights=[self.embedding_matrix],
                      input_length=self.sequence_len,
                      trainable=False)(words)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(units=self.lstm_units,
                               return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        result = Dense(self.num_class,
                       activation='sigmoid')(x)
        model = Model(inputs=words,
                      outputs=result)

        adam = optimizers.adam(lr=1e-3)
        model.compile(optimizer=adam,
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        model.summary()
        model.fit(np.array(self.train_seq),
                  np.array(self.train_label),
                  validation_data=(np.array(self.test_seq), np.array(self.test_label)),
                  batch_size=self.batch_size,
                  epochs=self.epoch,
                  callbacks=[checkpoint])

        test_label_loss, test_label_acc = model.evaluate(np.array(self.test_seq), np.array(self.test_label))
        print('test_label_acc:', test_label_acc)

        res = np.argmax(model.predict(self.test_seq), 1)
        # if not os.path.exists(lstm_model_file):
        #     model.save(lstm_model_file)
        return res

    #分类+实体识别模型
    def build_mul_model(self, train_ner, test_ner, ner_category):
        checkpoint = ModelCheckpoint(filepath=lstm_model_file, monitor='dense_acc', verbose=1, save_best_only=True)
        earlystop = EarlyStopping(monitor='dense_acc', patience=2)
        words = Input(shape=(None, ))
        x = Embedding(*self.embedding_matrix.shape,
                      weights=[self.embedding_matrix],
                      input_length=self.sequence_len,
                      trainable=False)(words)
        x = SpatialDropout1D(0.3)(x)
        x = Bidirectional(LSTM(units=self.lstm_units,
                               return_sequences=True))(x)
        crf = CRF(ner_category + 1,
                  learn_mode='join',
                  test_mode='viterbi',
                  sparse_target=False,
                  name='crf')
        pool = MaxPooling1D()(x)
        pool = Flatten()(pool)
        out1 = crf(x)
        out2 = Dense(self.num_class,
                     activation='sigmoid',
                     name='dense')(pool)
        model = Model(inputs=words,
                      outputs=[out1, out2])

        adam = optimizers.adam(lr=5e-3)
        model.compile(optimizer=adam,
                      loss={'crf': crf.loss_function, 'dense': 'categorical_crossentropy'},
                      loss_weights={'crf': 0.3, 'dense': 1},
                      metrics=['accuracy'])
        model.summary()
        model.fit(np.array(self.train_seq),
                  [np.array(train_ner), np.array(self.train_label)],
                  batch_size=self.batch_size,
                  epochs=self.epoch,
                  verbose=2,
                  callbacks=[checkpoint, earlystop])

        res_ner = []
        prediction_ner, prediction_label = model.predict(self.test_seq)[0], model.predict(self.test_seq)[1]
        for i in range(len(prediction_ner)):
            res_ner.append(np.argmax(prediction_ner[i], 1))
        res_label = np.argmax(prediction_label, 1)

        # if not os.path.exists(lstm_model_file):
        #     model.save(lstm_model_file)
        return np.array(res_ner), np.array(res_label)

    #实体识别模型
    def build_ner_model(self, train_ner, test_ner, ner_category):
        words = Input(shape=(None, ))
        x = Embedding(*self.embedding_matrix.shape,
                      weights=[self.embedding_matrix],
                      input_length=self.sequence_len,
                      trainable=False,
                      mask_zero=True)(words)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(units=self.lstm_units,
                               return_sequences=True))(x)
        crf = CRF(ner_category + 1,
                  learn_mode='join',
                  test_mode='viterbi',
                  sparse_target=False,
                  name='crf')
        model = crf(x)
        model = Model(inputs=words,
                      outputs=model)
        adam = optimizers.adam(lr=1e-3)
        model.compile(optimizer=adam,
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        model.summary()
        model.fit(np.array(self.train_seq),
                  np.array(train_ner),
                  batch_size=self.batch_size,
                  epochs=self.epoch,
                  callbacks=[self.checkpoint])

        test_label_loss, test_label_acc = model.evaluate(self.test_seq, np.array(test_ner))
        print('test_label_acc:', test_label_acc)

        res_ner = []
        prediction_ner = model.predict(self.test_seq)
        for i in range(len(prediction_ner)):
            res_ner.append(np.argmax(prediction_ner[i], 1))
        return np.array(res_ner)
