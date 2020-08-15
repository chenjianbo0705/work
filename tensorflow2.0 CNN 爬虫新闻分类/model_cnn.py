import  read_file
import  word2vec

import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os

import  numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

keji_data_file = "./data/keji.xlsx"
nba_data_file = "./data/NBA.xlsx"
nba2_data_file = './data/NBA2.xlsx'
x_train,y_train,x_test, y_test = read_file.load_positive_negative_data_files(keji_data_file, nba2_data_file,nba_data_file)

sentences, max_document_length_train = read_file.padding_sentences(x_train, '<PADDING>')
x_train = word2vec.embedding_sentences(sentences, embedding_size=32, min_count=1)
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = tf.squeeze(y_train, axis=1)

sentences, max_document_length_test = read_file.padding_sentences(x_test, '<PADDING>')
x_test = word2vec.embedding_sentences(sentences, embedding_size=32, min_count=1)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_test = tf.squeeze(y_test, axis=1)






def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x,y


train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.shuffle(50).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(32)


sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape)
def main():


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

    variables = conv_net.trainable_variables + fc_net.trainable_variables
    optimizer = optimizers.Adam(lr=0.0001)
    for epoch in range(1000):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_net(x)

                out = tf.reshape(out, [-1, 64])
                logist = fc_net(out)
                y_onehot = tf.one_hot(y, depth=2)

                loss = tf.losses.categorical_crossentropy(y_onehot, logist, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients((zip(grads, variables)))
            print(epoch, step, float(loss))
        total_num = 0
        total_correct = 0
        for x_test, y_test in test_db:

            out = conv_net(x_test)
            out = tf.reshape(out, [-1, 64])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x_test.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)
        fc_net.save_weights('./modle/weigths.cpkt')
        conv_net.save_weights('./model/con_ckpt')


if __name__ == '__main__':
    main()





