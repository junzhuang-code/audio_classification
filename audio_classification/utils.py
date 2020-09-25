#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: Utils Modules
@author: junzhuang
"""

import pandas as pd
import numpy as np
import pickle
import librosa
from librosa.effects import time_stretch, pitch_shift


def read_pickle(file_name):
    """Reload the dataset"""
    with open (file_name,'rb') as file:
        return pickle.load(file)

def dump_pickle(file_name, data):
    """Export the dataset"""
    with open (file_name,'wb') as file:
        pickle.dump(data, file)

def make_submission(Y_pred, name):#
    """make a submission: numpy to dataframe"""
    idx = [i for i in range(len(Y_pred))]
    submission = pd.DataFrame({'id': idx, 'label': Y_pred})
    submission.to_csv('{0}.csv'.format(name), index=False)

def gen_Y_pred(Y_pred_cnn, Y_pred_AS):
    """Generate final predicted label from Y_pred_AS"""
    assert len(Y_pred_cnn) == len(Y_pred_AS)
    # Modify the anomaly class in classification result.
    for i in range(len(Y_pred_AS)):
        if Y_pred_AS[i] == -1:
            Y_pred_cnn[i] = 2
    return Y_pred_cnn


# GMM ---
def flatten_3Dto2D(X):
    """Flatten the 3d testing set to 2d for clustering"""
    return np.array([X[i].flatten() for i in range(len(X))])

def compute_loss(X1, X2, p=2):
    """Compute L_{p} loss among two instances."""
    delta = X1 - X2
    delta_flat = delta.flatten()
    return np.linalg.norm(delta_flat, p)


# preprocessing ---
def add_white_noise(wave:np.array, delta:float=0.005) -> np.array:
    """
    @topic: Add white noise to audio wave.
    @inpit: wave: audio wave, delta: coefficient of white noise.
    @return: wave_wn: audio wave with white noise.    
    """
    white_noise = np.random.randn(len(wave))
    wave_wn = wave + delta*white_noise
    return wave_wn

def shift_wave(wave:np.array, pc_shift:float=0.05) -> np.array:
    """
    @topic: Shift the audio wave by given length.
    @inpit: wave: audio wave, pc_shift: the percentage of the length shifting.
    @return: wave_shift: shifted audio wave.    
    """
    shift_length = int(len(wave)*pc_shift)
    wave_shift = np.roll(wave, shift_length)
    return wave_shift

def stretch_wave(wave:np.array, speed_factor:float=0.8) -> np.array:
    """
    @topic: Change the speed of audio wave by given speed_factor.
    @inpit: wave: audio wave, speed_factor: the factor of speed stretch.
    @return: wave_stretch: stretched audio wave with the same length as original wave.
    """
    wave_stretch = time_stretch(wave, speed_factor)
    # pruning or padding the strethed wave
    if speed_factor < 1: # 音频长度被拉长
        wave_stretch = wave_stretch[:len(wave)] # 截取wave的同等长度
    elif speed_factor > 1: # 音频长度被缩短
        pad_zero = np.array([0. for _ in range(len(wave)-len(wave_stretch))])
        wave_stretch = np.hstack((wave_stretch, pad_zero)) # 后面补零
    assert len(wave_stretch) == len(wave)
    return wave_stretch

def shift_pitch(wave:np.array, sr:int, pitch_step:int=5) -> np.array:
    """
    @topic: Change the pitch of audio wave by given speed_factor.
    @inpit: wave: audio wave, sr: sampling rate, pitch_step: the step of pitch shift.
    @return: wave_sp: pitch shifted audio wave.
    """
    wave_sp = pitch_shift(wave, sr, pitch_step)
    return wave_sp

def MinMaxScaler(data, low, high):
    """
    @topic: Rescale 2D matrix into given ranges.
    @parameters: data (2d matrix), low/high (Scalar).
    @return: scaled data (2d matrix).
    """
    data_max, data_min = data.max(axis=0), data.min(axis=0)
    data_std = (data - data_min) / (data_max - data_min + 0.00001)
    data_scaled = data_std * (high - low) + low
    return data_scaled

def audio2spectrogram(wave:np.array, sr:int, n_mels:int=128) -> np.array:
    """
    @topic: Convert 1-D audio wave to 2-D spectrogram.
    @topic: wave: audio wave; sr: sample rate; n_mels: the size of output spectrogram.
    @return: 2-D spectrogram.
    @ref: https://blog.csdn.net/zzc15806/article/details/79603994
    """
    spec = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=n_mels)
    logspec = librosa.power_to_db(spec) # convert to log value
    return logspec
