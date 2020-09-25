#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: Anomaly Detection Using BiGAN and Generate Y_pred
@author: junzhuang
"""

import sys
import numpy as np
from utils import read_pickle, gen_Y_pred, make_submission
from model_BiGAN import BIGAN
import tensorflow as tf


# Initialize the arguments
try:
    NUM_EPOCHS = int(sys.argv[1])
    NUM_OUTLIERS = int(sys.argv[2])
    is_trainable = bool(sys.argv[3])
except:
    NUM_EPOCHS = 10
    NUM_OUTLIERS = 60 # Estimation of #outliers
    is_trainable = True
BATCH_SIZE = 32 # 64

# Read the pickle file
X_train = read_pickle('../audio_data/X_train4d.pkl')
X_test = read_pickle('../audio_data/X_test4d.pkl')
Y_train = read_pickle('../audio_data/Y_train1d.pkl')
print("The shape of X_train/X_test/Y_train: ", X_train.shape, X_test.shape, Y_train.shape)

# Instantiate the model
bigan = BIGAN(X_train.shape[1], X_train.shape[2], X_train.shape[3])

if is_trainable:
    # Training the BiGAN
    bigan.train_by_batch(X_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    #bilstm.train_all(X_train_, Y_train_, BATCH_SIZE, NUM_EPOCHS)
else:
    # Restore the checkpoint
    checkpoint_dir = './runs/checkpoint_bigan'
    checkpoint = tf.train.Checkpoint()
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    print("Checkpoint restored for Anomaly Detection!")

    # Anomaly Detection
    AS = bigan.compute_anomaly_score(X_train, Y_train, X_test)
    # Prediction
    ts = NUM_OUTLIERS/len(X_test) # Find out the best threshold
    Y_pred_AS = bigan.predict_outlier(AS, ts)
    #print("Y_pred_AS: ", Counter(Y_pred_AS))

    # Geneate final Y_pred and make submission
    Y_pred = np.load('Y_pred.npy')
    Y_pred_new = gen_Y_pred(Y_pred, Y_pred_AS)
    print("Y_pred_new.shape: ", Y_pred_new.shape)
    make_submission(Y_pred_new, "submission")
