#!/usr/bin/env python

# Character level "Vanilla" RNN
# conceptually based on https://gist.github.com/karpathy/d4dee566867f8291f086

import tensorflow as tf
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder
from tqdm import *

# Hyperparameters
RNN_NUM_UNITS = 100
BATCHSIZE = RESET_NUM = 25
TIMESTEPS = 1

# Load input txt file

def load_data():
    data = open(sys.argv[1], 'r').read()
    chars = list(set(data))
    # Assign labels to vocab characters
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    # Reverse procedure
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    return data, chars, char_to_ix, ix_to_char

# Control run flow

def main():
    # Load data
    [data, vocab, c2i, i2c] = load_data()
    vocab_size = len(vocab)
    # Init Placeholders and I/O of VRNN
    X = tf.placeholder("float", shape=[None, RESET_NUM, vocab_size])
    seqY = tf.placeholder("float", shape=[None, RESET_NUM, vocab_size])
    seqX = tf.unstack(X, RESET_NUM, 1)
    # Init VRNN
    vrnn_cell = tf.contrib.rnn.BasicRNNCell(RNN_NUM_UNITS)
    outputs, states = tf.nn.static_rnn(vrnn_cell, seqX, dtype=tf.float32)
    # Output
    out = tf.layers.dense(outputs[-1], vocab_size, activation=None)
    # Loss operator
    loss_op = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=out, labels=seqY)))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        ohe = OneHotEncoder(n_values=vocab_size, sparse=False)
        for batch in tqdm(xrange(0, len(data), BATCHSIZE), ascii=True):
            idxIn = slice(batch, batch + BATCHSIZE)
            idxOut = slice(batch+1, batch+1 + BATCHSIZE)
            seqIn = [c2i[ch] for ch in data[idxIn]]
            seqOut = [c2i[ch] for ch in data[idxOut]]
            seqIn = np.array(seqIn).reshape(BATCHSIZE, 1)
            seqOut = np.array(seqOut).reshape(BATCHSIZE, 1)
            sess.run(train_op, feed_dict={X: ohe.fit_transform(seqIn).reshape(1, BATCHSIZE, vocab_size), seqY: ohe.fit_transform(seqOut).reshape(1, BATCHSIZE, vocab_size)})


if __name__ == "__main__":
    main()
