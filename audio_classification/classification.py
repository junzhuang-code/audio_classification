#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: Audio Classification by 2-D CNN
@author: junzhuang
"""

import sys
import numpy as np
from utils import read_pickle
from sklearn.model_selection import train_test_split
from model_CNN import CNN
from model_LSTM import BiLSTM
from collections import Counter

# Initialize the arguments
try:
    data_type = str(sys.argv[1])
    NUM_EPOCHS = int(sys.argv[1])
except:
    data_type = "4d" # 3d for lstm, 4d for cnn/gan.    
    NUM_EPOCHS = 100
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# Import dataset
Y_train = read_pickle('../audio_data/Y_train1d.pkl')
if data_type == "3d":
    X_train = read_pickle('../audio_data/X_train3d.pkl')
    X_test = read_pickle('../audio_data/X_test3d.pkl')
    assert len(X_train.shape) == 3
if data_type == "4d":
    X_train = read_pickle('../audio_data/X_train4d.pkl')
    X_test = read_pickle('../audio_data/X_test4d.pkl')
    assert len(X_train.shape) == 4
    n_channels = X_train.shape[3]
print("The shape of {0} X_train/X_test/Y_train: ".format(data_type), \
          X_train.shape, X_test.shape, Y_train.shape)
n_rows, n_cols, n_classes = X_train.shape[1], X_train.shape[2], Y_train.max()+1

# Split dataset for evaluation
X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)
print(X_train_.shape, X_test_.shape)

if data_type == "3d": # Train BiLSTM model
    model = BiLSTM(n_rows, n_cols, n_classes, LEARNING_RATE)
    model.train_by_batch(X_train_, Y_train_, BATCH_SIZE, NUM_EPOCHS)
    #model.train_all(X_train_, Y_train_, BATCH_SIZE, NUM_EPOCHS)
if data_type == "4d": # Train CNN model
    model = CNN(n_rows, n_cols, n_channels, n_classes, LEARNING_RATE)
    model.train(X_train_, Y_train_, BATCH_SIZE, NUM_EPOCHS)

# Generate predicted label
Y_pred = model.prediction(X_test) # for submission
print("The shape of Y_pred: ", Y_pred.shape)
np.save("Y_pred.npy", Y_pred) # Save Y_pred

# Evaluation on labeled dataset
model.evaluation(X_test_, Y_test_)
print("Y_test_: ", Counter(Y_test_))
print("Y_pred: ", Counter(Y_pred.numpy()))
