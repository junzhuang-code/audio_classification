#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: Preprocessing
@author: junzhuang
@ref:
    https://www.kaggle.com/CVxTz/audio-data-augmentation
    https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
    https://medium.com/@keur.plkar/audio-data-augmentation-in-python-a91600613e47
"""

import sys
import numpy as np
import pandas as pd
from utils import add_white_noise, shift_wave, stretch_wave, shift_pitch, \
                    MinMaxScaler, audio2spectrogram, dump_pickle
from collections import Counter


def audio_data_augmentation(data, label, sample_rate):
    """
    @topic: Implement data augmentation on audio dataset by adforementioned methods.
    @input: data/label: dataset and label; sample_rate: sample rate.
    @return: Augmented datset and label.
    """
    data_aug, label_aug = [], []
    for i in range(len(data)):
        # augment the data by different methods
        data_aug.append(add_white_noise(data[i], delta=0.005))
        data_aug.append(add_white_noise(data[i], delta=0.01))
        data_aug.append(shift_wave(data[i], pc_shift=0.05))
        data_aug.append(shift_wave(data[i], pc_shift=0.1))
        data_aug.append(stretch_wave(data[i], speed_factor=0.7))
        data_aug.append(stretch_wave(data[i], speed_factor=1.3))
        data_aug.append(shift_pitch(data[i], sr=sample_rate, pitch_step=-5))
        data_aug.append(shift_pitch(data[i], sr=sample_rate, pitch_step=5))
        label_aug.extend([label[i] for _ in range(8)]) # append the label
    data_new, label_new = np.vstack((data, data_aug)), np.hstack((label, label_aug)) # merge the dataset
    idx_new = np.arange(len(data_new)) # get the index of new dataset
    np.random.shuffle(idx_new) # shuffle the index
    return data_new[idx_new], label_new[idx_new]

def get_spectrogram_data(data:np.array, sr:int, n_mels:int=32, types:str="4d") -> np.array:
    """
    @topic: Convert 1-D audio wave to 2-D spectrogram and build 4-D dataset.
    @topic: wave: audio wave; sr: sample rate; n_mels: the size of output spectrogram; types: type of data.
    @return: 4-D spectrogram dataset (n_samples, rows, cols, channels).
    """
    data_sp = []
    for i in range(len(data)):
        logspec_i = audio2spectrogram(data[i], sr, n_mels)
        logspec_i_crop = logspec_i[0:n_mels, 0:n_mels]        
        data_sp.append(logspec_i_crop)
    if types == "4d":
        data_sp = np.expand_dims(np.array(data_sp), axis=-1) # add 1 dim
    if types == "3d":
        data_sp = np.array(data_sp)
    return data_sp.astype("float32")


if __name__ == "__main__":
    # Initialize the arguments
    try:
        data_type = str(sys.argv[1])
    except:
        data_type = "3d" # 3d for lstm, 4d for cnn/gan.

    # Import audio dataset
    X_train = np.load("../audio_data/audio_train.npy")
    X_test = np.load("../audio_data/audio_test.npy")
    Y_train = pd.read_csv("../audio_data/labels_train.csv")
    Y_train = np.array(Y_train)[:, -1] # set(Y_train) = {0, 1, 3, 4, 5, 6, 7, 8, 9}
    Counter(Y_train)

    # Audio Data Augmentation
    sample_rate = 20050
    X_aug, Y_aug = audio_data_augmentation(X_train, Y_train, sample_rate)
    print("The shape of X_aug/Y_aug: ", X_aug.shape, Y_aug.shape)

    # Convert training/testing set into 2-D spectrogram
    X_train_sp = get_spectrogram_data(X_aug, sample_rate, n_mels=32, types=data_type)
    X_train_scaled = MinMaxScaler(X_train_sp, -1, 1)
    X_test_sp = get_spectrogram_data(X_test, sample_rate, n_mels=32, types=data_type)
    X_test_scaled = MinMaxScaler(X_test_sp, -1, 1)
    print("The shape of X_train_scaled/X_test_scaled: ", X_train_scaled.shape, X_test_scaled.shape)
    print("Preprocessing Done!")    

    # Dump to pickle file
    dump_pickle('../audio_data/X_train{0}.pkl'.format(data_type), X_train_scaled)
    dump_pickle('../audio_data/X_test{0}.pkl'.format(data_type), X_test_scaled)
    dump_pickle('../audio_data/Y_train1d.pkl', Y_aug)
