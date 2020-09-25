#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: Test 1-D CNN model on competition dataset
@author: junzhuang
@ref:
    https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
"""

from tensorflow.keras import layers, models, callbacks, backend
import numpy as np
import pandas as pd
import sys


# Build 1-D CNN model
def CNN_1d(input_shape, n_classes, opt='adam'):
    # 1-D CNN model
    # input: input_shape=(n_features, 1), n_classes=int, opt="sgd"or"adam".
    # Build the model
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))
    # Compile the model
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Make submission
def make_submission(Y_pred, name):
    # make a submission: numpy to dataframe
    idx = [i for i in range(len(Y_pred))]
    submission = pd.DataFrame({'id': idx, 'label': Y_pred})
    submission.to_csv('{0}.csv'.format(name), index=False)


if __name__ == "__main__":
    # Initialize the arguments
    try:
        NUM_EPOCHS = int(sys.argv[1])
    except:
         NUM_EPOCHS = 1
    BATCH_SIZE = 32

    # Import audio dataset
    X_train = np.load("../audio_data/audio_train.npy")
    X_test = np.load("../audio_data/audio_test.npy")
    Y_train = pd.read_csv("../audio_data/labels_train.csv")

    # Simple preprocessing
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_train = np.array(Y_train)[:, -1] # set(Y_train) = {0, 1, 3, 4, 5, 6, 7, 8, 9}
    # Y_train = utils.to_categorical(Y_train) for "categorical_crossentropy" loss.
    print("The shape of X_train\X_test\Y_train: ", X_train.shape, X_test.shape, Y_train.shape)

    # Build 1-D CNN
    input_shape = (X_train.shape[1], 1)
    n_classes = int(Y_train.max()+1) # 10
    model = CNN_1d(input_shape, n_classes, opt='adam')
    model.summary()

    # Train the model
    tbCallBack = callbacks.TensorBoard(log_dir="./runs/TB_cnn1d")
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                            validation_split=0.2, callbacks=[tbCallBack])
    #model.save("./runs/checkpoint_cnn1d")

    # Generate predicted label
    #models.load_model("./runs/checkpoint_cnn1d")
    prob_table = model.predict(X_test)
    Y_pred = backend.argmax(prob_table)
    print("The shape of Y_pred: ", Y_pred.shape)

    # Make submission file
    make_submission(Y_pred, "submission")
