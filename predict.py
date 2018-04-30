# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function
import glob
import argparse
import sys
import os.path
import pickle
import time

from scipy import signal
import librosa
import numpy as np
import tensorflow as tf

import hyperparams as hp
from hyperparams import logdir_path

import matplotlib.pyplot as plt


num_classes = 61
num_features = 40


# HYPER PARAMETERS
TRAIN_CAP = 1000
TEST_CAP = 500
NUM_LAYERS = 3
NUM_HIDDEN = 150
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 50
KEEP_PROB = 0.6


SAVE_DIR = "./checkpoint/save"
PLOTTING = True

SAVE_PER_EPOCHS = 1
SHUFFLE_PER_EPOCHS = 20


def preemphasis(x, coeff=0.97):
    '''
    Applies a pre-emphasis filter on x
    '''
    return signal.lfilter([1, -coeff], [1], x)


def load_vocab():
    '''
    Returns:
    phn2idx - A dictionary containing phoneme string to index mappings
    idx2phn - A dictionary containing index to phoneme mappings (reverse of phn2idx)
    '''
    phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
            'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
            'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
            'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
            'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    # Phoneme to index mapping
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    # Index to phoneme mapping
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn


def _get_mfcc_log_spec_and_log_mel_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):
    '''
    Args:
    wav - Wave object loaded using librosa

    Returns:
    mfcc - coefficients
    mag - magnitude spectrum
    mel
    '''
    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft,
                     hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(
        hp.Default.sr, hp.Default.n_fft, hp.Default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs
    db = librosa.amplitude_to_db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.Default.n_mfcc, db.shape[0]), db)

    # Log
    mag = np.log(mag + sys.float_info.epsilon)
    mel = np.log(mel + sys.float_info.epsilon)

    # Normalization
    # self.y_log_spec = (y_log_spec - hp.mean_log_spec) / hp.std_log_spec
    # self.y_log_spec = (y_log_spec - hp.min_log_spec) / (hp.max_log_spec - hp.min_log_spec)

    return mfccs.T, mag.T, mel.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


def load_test_data(wav_file):
    mfccs, _, _ = _get_mfcc_log_spec_and_log_mel_spec(wav_file, hp.Default.preemphasis, hp.Default.n_fft,
                                                      hp.Default.win_length,
                                                      hp.Default.hop_length)
    print("Loaded mfccs from PREDICT data")
    return np.array([mfccs])


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('predict_file', type=str, help='predict file path')
    arguments = parser.parse_args()
    return arguments


def one_hot(indices, depth=num_classes):
    one_hot_labels = np.zeros((len(indices), depth))
    one_hot_labels[np.arange(len(indices)), indices] = 1
    return one_hot_labels


def predict_phonemes(predict_file):
    graph = tf.Graph()
    with graph.as_default():
        # Input placeholder of shape [BATCH_SIZE, num_frames, num_mfcc_features]
        inputs = tf.placeholder(tf.float32, [None, None, num_features])

        # Target placeholder of shape [BATCH_SIZE, num_frames, num_phn_classes]
        targets = tf.placeholder(tf.int32, [None, None, num_classes])

        # List of sequence lengths (num_frames)
        seq_len = tf.placeholder(tf.int32, [None])

        keep_prob = tf.placeholder(tf.float32, shape=())

        # Get a GRU cell with dropout for use in RNN
        def get_a_cell(gru_size, keep_prob=1.0):
            gru = tf.nn.rnn_cell.BasicLSTMCell(gru_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(
                gru, output_keep_prob=keep_prob)
            return drop

        # Make a multi layer RNN of NUM_LAYERS layers of cells
        stack = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(NUM_HIDDEN, keep_prob) for _ in range(NUM_LAYERS)])

        # outputs is the output of the RNN at each time step (frame)
        # RNN has NUM_HIDDEN output nodes
        # outputs has shape [BATCH_SIZE, num_frames, NUM_HIDDEN]
        # The second output is the last state and we will not use that
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            stack, stack, inputs, seq_len, dtype=tf.float32)
        outputs = tf.concat([output_fw, output_bw], axis=2)

        # Save input shape for restoring later
        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        # outputs is now of shape [BATCH_SIZE*num_frames, NUM_HIDDEN]
        # So the same weights are trained for each timestep of each sequence
        outputs = tf.reshape(outputs, [-1, 2 * NUM_HIDDEN])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([2 * NUM_HIDDEN,
                                             num_classes],
                                            stddev=0.1))
        # Zero initialization
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        # logits = tf.transpose(logits, (1, 0, 2))

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=targets))

        optimizer = tf.train.AdamOptimizer(
            LEARNING_RATE).minimize(cross_entropy)

        # define an accuracy assessment operation
        correct_prediction = tf.equal(
            tf.argmax(logits, 2), tf.argmax(targets, 2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        SAVE_PATH = SAVE_DIR + '_{}_{}_{}_{}/model.ckpt'.format(
            NUM_HIDDEN, NUM_LAYERS, LEARNING_RATE, BATCH_SIZE)
        try:
            saver.restore(sess, SAVE_PATH)
            print("Model restored.\n")
        except:
            print("Cannot load model. First train it.\n")
            exit()

        wav, sr = librosa.load(predict_file, sr=hp.Default.sr)
        predict_inputs = load_test_data(wav)

        predict_inputs = (predict_inputs - 26.4842046909) / \
            19.3570720935

        num_examples = len(predict_inputs)

        predict_seq_len = [len(x) for x in predict_inputs]

        feed = {inputs: predict_inputs,
                seq_len: predict_seq_len,
                keep_prob: 1.0}

        outputs = sess.run(logits, feed)

        outputs = [np.argmax(out, axis=1) for out in outputs]

        return outputs


if __name__ == '__main__':
    args = get_arguments()
    predict_file = args.predict_file

    outputs = predict_phonemes(predict_file)
    print(outputs)

    phn2idx, idx2phn = load_vocab()
    phns = [idx2phn[x] for x in outputs[0]]
    # prev = 'a'
    # string = ''
    # for phn in phns:
    #     if phn != prev:
    #         string += phn + ' '
    #         prev = phn
    # print(string)

    print(phns)
