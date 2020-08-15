import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers
from  data_util import  *
from  config import Config


import  random



class MyRNN(keras.Model):

    def __init__(self, ):
        super(MyRNN, self).__init__()


        # [b, 8, 804] ,
        self.rnn = keras.Sequential([
            layers.SimpleRNN(521, dropout=0.5, return_sequences=True, unroll=True),
            layers.SimpleRNN(804, dropout=0.5, unroll=True)
        ])


        self.outlayer = layers.Dense(804)

    def call(self, inputs, training=None):


        x = inputs

        x = self.rnn(x)

        x = self.outlayer(x)

        return x


def main():
    epochs = 1

    x,y= preprocess_file(Config)
    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_train = db_train.shuffle(1000).batch(64, drop_remainder=True)

    model = MyRNN()
    model.compile(optimizer = keras.optimizers.Adam(0.001),
                  loss = tf.losses.BinaryCrossentropy())
    model.fit(db_train, epochs=epochs)
    model.save_weights('./model/model.model')


if __name__ == '__main__':
   main()


