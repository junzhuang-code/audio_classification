#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: CNN model
@author: junzhuang
@ref:
    https://www.tensorflow.org/tutorials/images/cnn
"""

from tensorflow.keras import models, layers, callbacks, optimizers, losses, backend

## Build 2-D CNN model
class CNN():
    """Convolutional neural networks."""
    def __init__(self, n_rows, n_cols, n_channels, n_classes, LEARNING_RATE=0.01):
        self.rows = n_rows
        self.cols = n_cols
        self.channels = n_channels
        self.classes = n_classes
        self.lr = LEARNING_RATE
        self.checkpoint_path = './runs/checkpoint_cnn2d'
        self.tb_path = './runs/TB_cnn2d'
        # Initialize CNN model
        self.model = self.cnn_model()
        self.model.compile(optimizer=optimizers.Adam(self.lr),
                          loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

    def cnn_model(self):
        # Build 2-D CNN model
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', \
                                input_shape=(self.rows, self.cols, self.channels)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.classes))
        model.summary()
        return model

    def train(self, X_train, Y_train, BATCH_SIZE, NUM_EPOCHS):
        # Setup the callback function
        tb = callbacks.TensorBoard(log_dir=self.tb_path)
        earlystop = callbacks.EarlyStopping(monitor='loss', patience=5)
        checkpoint = callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                save_weights_only=True,
                                                monitor='val_acc',
                                                save_best_only=True,
                                                mode='max')
        # Fitting the model
        self.model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, shuffle=True, \
                      verbose=1, validation_split=0.2, callbacks=[tb, checkpoint, earlystop])
        self.model.save(self.checkpoint_path)

    def prediction(self, X):
        # Prediction
        prob_table = self.model.predict(X)
        Y_pred = backend.argmax(prob_table)
        return Y_pred

    def evaluation(self, X, Y):
        # Evaluation
        self.model.evaluate(X, Y, verbose=2)
