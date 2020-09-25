#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: LSTM model
@author: junzhuang
@ref:
    https://www.kaggle.com/carlolepelaars/bidirectional-lstm-for-audio-labeling-with-keras#Birectional-LSTM-model-for-audio-labeling-with-Keras
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, backend, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

class BiLSTM():
    def __init__(self, n_rows, n_cols, n_classes, LEARNING_RATE):
        self.rows = n_rows # 32
        self.cols = n_cols # 32
        self.classes = n_classes
        self.lr = LEARNING_RATE # 1e-3
        #self.opt = optimizers.Adam(LEARNING_RATE, decay=1e-5)
        self.opt = optimizers.Adam(5*LEARNING_RATE, beta_1=0.1, beta_2=0.001, amsgrad=True)
        self.checkpoint_path = './runs/checkpoint_bilstm'
        self.tb_path = './runs/TB_bilstm'
        # Initialize LSTM model
        self.model = self.bilstm_model()
        self.model.compile(optimizer=self.opt,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def bilstm_model(self):
        if tf.test.is_gpu_available():
            lstm_model = CuDNNLSTM
        else:
            lstm_model = LSTM
        model = Sequential()
        model.add(Bidirectional(lstm_model(128, return_sequences=True, \
                                    input_shape=(self.rows, self.cols))))
        model.add(Dropout(0.2))
        model.add(lstm_model(128))
        #model.add(LSTM(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.classes, activation='softmax'))
        return model

    def train_by_batch(self, X_train, Y_train, BATCH_SIZE, NUM_EPOCHS):
        # Log for TensorBoard
        summary_writer = tf.summary.create_file_writer(self.tb_path)
        # Fitting the model
        n_batch = len(X_train)//BATCH_SIZE
        for epoch in range(NUM_EPOCHS):
            loss_list, acc_list = [], []
            for b in range(n_batch):
                # Decide the range of one batch
                start = b * BATCH_SIZE
                if b == n_batch - 1:
                    end = len(X_train)
                else:
                    end = start + BATCH_SIZE
                X_batch = X_train[start:end, :]
                Y_batch = Y_train[start:end]
                loss, acc = self.model.train_on_batch(X_batch, Y_batch)
                loss_list.append(loss)
                acc_list.append(acc)
                # Log the progress of each batch
                with summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=b)
                    tf.summary.scalar('acc', acc, step=b)
            # Plot the progress
            print("No.{0}: loss: {1}, acc: {2}.".\
                  format(epoch, np.mean(loss_list), np.mean(acc_list)))
        # Save the model
        self.model.save(self.checkpoint_path)

    def train_all(self, X_train, Y_train, BATCH_SIZE, NUM_EPOCHS):
        tb = callbacks.TensorBoard(log_dir=self.tb_path)
        es = callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
        self.model.fit(X_train, Y_train, BATCH_SIZE, NUM_EPOCHS, shuffle=True, \
                         verbose=1, validation_split=0.2, callbacks = [tb, es])

    def prediction(self, X):
        # Prediction
        prob_table = self.model.predict(X)
        Y_pred = backend.argmax(prob_table)
        return Y_pred

    def evaluation(self, X, Y):
        # Evaluation
        self.model.evaluate(X, Y, verbose=2)
