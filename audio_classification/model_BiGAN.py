#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: CS590 Competition -- Audio Classification
@topic: BiGAN for Anomaly Detection
@author: junzhuang
@ref:
    github: https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
    tutorial: https://www.tensorflow.org/tutorials/generative/dcgan?hl=zh-cn
"""

import os
import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, \
    MaxPooling2D, Flatten, Dropout, BatchNormalization, concatenate, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class BIGAN():
    def __init__(self, img_rows, img_cols, channels):
        self.img_rows = img_rows # 32
        self.img_cols = img_cols # 32
        self.channels = channels # 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Optimizer
        self.lr = 1e-2
        self.optimizer = Adam(self.lr, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                     optimizer=self.optimizer)

    def build_encoder0(self): # 临时弃用
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(256))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.latent_dim))
        model.summary()
        x_real = Input(shape=self.img_shape)
        z_ = model(x_real)
        return Model(x_real, z_)

    def build_encoder(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.img_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.latent_dim))
        model.summary()
        x_real = Input(shape=self.img_shape)
        z_ = model(x_real)
        return Model(x_real, z_)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(8*8*256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8, 8, 256)))
        assert model.output_shape == (None, 8, 8, 256) # 注意：batch size 没有限制
        model.add(Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 256)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        #assert model.output_shape == (None, 32, 32, 64)
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.img_rows, self.img_cols, self.channels)
        model.summary()
        z = Input(shape=(self.latent_dim,))
        x_rec = model(z)
        return Model(z, x_rec)

    def build_discriminator(self):
        z = Input(shape=(self.latent_dim,)) # Latent variable
        x = Input(shape=self.img_shape) # Samples
        d_in = concatenate([z, Flatten()(x)]) # Concatenate z and x
        # Model output
        model = Dense(128, kernel_initializer = 'he_normal')(d_in)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        #model = Dense(128)(model)
        #model = LeakyReLU(alpha=0.2)(model)
        #model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)
        return Model(inputs=[z, x], outputs=validity)

    def train(self, X_train, epochs=100, batch_size=128):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Log for TensorBoard
        log_dir = "./runs/TB_bigan"
        summary_writer = tf.summary.create_file_writer(log_dir)

        # Initialize the checkpoint
        interval = int(epochs//10) if epochs >= 10 else 5
        checkpoint_dir = './runs/checkpoint_bigan'
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        # Restore the latest checkpoint in checkpoint_dir
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        x_rec_list = []
        for epoch in range(epochs):
            # Sample random noise z
            z = np.random.normal(size=(batch_size, self.latent_dim)) # 随机生成噪音

            # generate fake data point x
            x_fake = self.generator.predict(z) # 以上述隐变量z生成假数据点x_fake

            # Select a random batch of data point x and encode
            idx = np.random.randint(0, X_train.shape[0], batch_size) # len(idx)=batch_size
            x_real = X_train[idx] # size = batch_size x (img_rows, img_cols, 1)
            z_ = self.encoder.predict(x_real) # 随机选一批真数据点x_real进行编码，生成一批潜变量z_ (batch_size x latent_dim) 

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, x_real], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, x_fake], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator (z -> x_real is valid, & x_real -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, x_real], [valid, fake])

            # For Experiments and Visualization ---------------------
            # Save scalars into TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('D_loss', d_loss[0], step=epoch)
                tf.summary.scalar('G_loss', g_loss[0], step=epoch)
                tf.summary.scalar('D_acc', 100*d_loss[1], step=epoch)
                tf.summary.scalar('G_acc', 100*g_loss[1], step=epoch)

            # Save the checkpoint at given interval
            if (epoch + 1) >= int(interval) and (epoch + 1) % int(interval) == 0:
                checkpoint.save(file_prefix = str(checkpoint_path))

            # Store the reconstructed samples
            if (epoch + 1) >= int(interval) and (epoch + 1) % int(interval) == 0:
                    x_rec = self.predict_x_rec(num_img=5)
                    x_rec_list.append(x_rec)

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, acc: %.2f%%]"\
                   % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]))

        # Save file
        np.save("reconstructed_samples.npy", x_rec_list)  
        checkpoint.save(file_prefix = str(checkpoint_path))

    def predict_x_rec(self, num_img=1):
        # Store the reconstructed samples at specific epoch
        z = np.random.normal(size=(int(num_img), self.latent_dim))
        #z = self.encoder.predict(X) # generated latent variable
        x_rec = self.generator.predict(z) # generated x from z
        #x_rec = 127.5 * x_rec + 127.5
        return x_rec

    def plot_x_rec(self, x_rec_list):
        # Plot all reconstructed samples along epochs
        # input: x_rec_list size=(num_sampling, num_img, rows, cols, dim).
        cur_path = os.getcwd()
        if not os.path.exists('{0}/Images'.format(cur_path)):
            os.makedirs('Images')
        plt.figure(figsize=(5,5))
        for s in range(len(x_rec_list)):
            for i in range(x_rec_list[s].shape[0]):
                plt.subplot(1, x_rec_list[s].shape[0], i+1)
                x_rec_s_i = np.squeeze(x_rec_list[s][i], axis=2)
                plt.imshow(x_rec_s_i, interpolation='nearest', cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig("Images/Image_{0}.png".format(s))
            plt.close()

    def split_data(self, X, Y, target):
        """Split the data with given target(label#)"""
        X_sp, Y_sp = [], []
        for i, label in enumerate(Y):
            if label == target:
                X_sp.append(X[i])
                Y_sp.append(Y[i])
        return np.array(X_sp), np.array(Y_sp)

    def compute_rec_loss(self, X):
        """
        @topic: Compute the loss between real instance and reconstructed instance.
        @input: X (4D): real instance.
        @return: array of loss between x and G(E(x)).
        """
        z_ = self.encoder.predict(X) # generated latent variable
        x_rec = self.generator.predict(z_) # generated x from z_
        loss_list = []
        for i in range(0, len(X)):
            delta = X[i] - x_rec[i] # delta = data point - reconstructed point
            delta_flat = delta.flatten() # or = np.ndarray.flatten(delta)
            rec_loss = np.linalg.norm(delta_flat) # compute L2 norm
            #rec_loss = np.linalg.norm(delta_flat, 1) # compute L1 norm
            loss_list.append(rec_loss)
        return np.array(loss_list)

    def compute_anomaly_score(self, X_train, Y_train, X_test):
        """
        @topic: Compute Anomaly_Score (AS).
        @input: X_train, X_test (2D); Y_train (1D).
        @return: array of Anomaly_Score.
        """
        # 计算训练集里K个簇的loss的中位数
        K_list = list(set(Y_train))
        L_train_k_list = [] # len = K
        for k in range(len(K_list)): 
            X_train_k, _ = self.split_data(X_train, Y_train, Y_train[k]) # 分离第k个簇
            L_train_k = self.compute_rec_loss(X_train_k) # 计算该簇的loss
            L_train_k_median = np.median(L_train_k) # 取L_train_k的中位数作为基准
            L_train_k_list.append(L_train_k_median)
        # 计算 Anomaly_Score
        assert len(X_test.shape) == 4
        L_test = self.compute_rec_loss(X_test) # 计算测试集的重构loss
        AS = []
        for i in range(len(X_test)): # 遍历所有的测试点
            AS_k_list = []
            for k in range(len(L_train_k_list)):
                AS_k = abs(L_test[i] - L_train_k_list[k]) # 计算该测试点到第k个簇的AS
                AS_k_list.append(AS_k)
            AS_i = min(AS_k_list) # 取k个簇的最小AS作为该点的AS。
            AS.append(AS_i)
        # 如果测试点的loss与基准之差比较大（即两个loss很不一样），说明这很大可能是异常值。
        return np.array(AS)

    def predict_outlier(self, AS, ts):
        """
        @topic: Predict the outlier based on UCS.
        @input: AS (vector), given threshold (float).
        @return: Y_pred: the predicted labels based on the UC Score.
        """
        ts_idx = int(len(AS)*(1 - ts)) # the cut-off index
        AS_copy = AS.copy()
        AS_copy.sort() # sorting as increasing order
        ts_AS = AS_copy[ts_idx] # the cut-off score
        Y_pred = []
        for i in range(len(AS)):
            if AS[i] >= ts_AS:  # 大于阈值的则是异常值
                Y_pred.append(-1)
            else:
                Y_pred.append(0)
        return Y_pred
