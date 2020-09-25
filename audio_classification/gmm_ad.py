#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: Anomaly Detection Using GMM and Generate Y_pred
@author: junzhuang
"""

import sys
import numpy as np
from utils import read_pickle, flatten_3Dto2D, compute_loss, make_submission
from sklearn.mixture import GaussianMixture


def GMM(X, num_class=10):
    """Build GMM model"""
    gmm = GaussianMixture(n_components=num_class, random_state=11)
    gmm.fit(X)
    return gmm.predict(X), gmm.means_, gmm.covariances_

def find_close_point(X, Y, center, TARGET=2, k=50):
    """
    @topic: Find top k cloest points to given center.
    @input: 
        X: dataset(2D); Y: label(1D);
        center: the center of target cluster;
        TARGET: the target number that we need to handle;
        k: top k cloest points.
    @return: the index of top k cloest points to given center in lable.
    """
    loss_arr = np.zeros_like(Y, dtype=float)
    for l in range(len(Y)):
        if Y[l] == TARGET:
            loss = compute_loss(X[l], center)
            loss_arr[l] = loss
        else:
            loss_arr[l] = float("inf")
    return np.array(loss_arr).argsort()[:k]

def gen_Y_pred_AD(Y_pred, target_idx, TARGET):
    """
    @topic: Generate Y_pred after anomaly detection with target_idx list
    @input:
        Y_pred: predicted label by classification;
        target_idx: the list of target number;
        TARGET: target number.
    @return: the new Y_pred.
    """
    Y_pred[target_idx] = TARGET
    return Y_pred


if __name__ == "__main__":
    # Initialize the arguments
    try:
        k = int(sys.argv[1])
        TARGET = int(sys.argv[2])
        NUM_CLASSES = int(sys.argv[3])
    except:
        k=60
        TARGET=2
        NUM_CLASSES = 10
    # Read the pickle file
    X_test = read_pickle('../audio_data/X_test3d.pkl')
    Y_pred = np.load('Y_pred.npy')
    Y_pred = np.array(Y_pred)[:, -1]
    # Flatten 3d testing set to 2d.
    X_test2d = flatten_3Dto2D(X_test)
    # Employ GMM to cluster digit 2.
    Y_pred_test, Y_pred_mu, Y_pred_cov = GMM(X_test2d, num_class=NUM_CLASSES)
    # Select the points that are close to the center of digit 2.
    center = Y_pred_mu[1]
    min_k_idx = find_close_point(X_test2d, Y_pred_test, center, TARGET, k)
    print("The list of predicted target digit: ", min_k_idx)
    # Generate new Y_pred by given target index list
    Y_pred_new = gen_Y_pred_AD(Y_pred, min_k_idx, TARGET)
    print("New Y_pred: ", Y_pred_new)
    make_submission(Y_pred_new, "submission")
