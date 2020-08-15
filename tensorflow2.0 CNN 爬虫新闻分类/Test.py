import  read_file
import  word2vec

import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os

import  numpy as np

x = tf.random.normal((1, 4592, 32))

conv_layers = [tf.keras.layers.Conv1D(64, 4, strides=2, activation='relu', padding="same"),
                   tf.keras.layers.Conv1D(64, 4, strides=2, activation='relu', padding="same"),
                   tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),

                   tf.keras.layers.Conv1D(64, 4, strides=2, activation='relu', padding="same"),
                   tf.keras.layers.Conv1D(64, 4, strides=2, activation='relu', padding="same"),
                   tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),

                   tf.keras.layers.Conv1D(64, 4, strides=2, activation='relu', padding="same"),
                   tf.keras.layers.Conv1D(64, 4, strides=2, activation='relu', padding="same"),
                   tf.keras.layers.MaxPooling1D(pool_size=1, strides=72, padding='same'),
                   ]
conv_net = Sequential(conv_layers)

fc_net = Sequential([

        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(2, activation=tf.nn.relu),
    ])
conv_net.build(input_shape=[None, 5942, 32])
fc_net.build(input_shape=[None, 64])
conv_net.load_weights('con_ckpt')
fc_net.load_weights('weigths.cpkt')


out = conv_net(x)

out = tf.reshape(out, [-1, 64])
logist = fc_net(out)
prob = tf.nn.softmax(logist, axis=1)
#pred = tf.argmax(prob, axis=1)
#pred = tf.cast(pred, dtype=tf.int32)
print(prob)


