# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function
import glob
import argparse
import sys
import os.path
import time

from scipy import signal
import librosa
import numpy as np
import tensorflow as tf

import hyperparams as hp

import matplotlib.pyplot as plt


num_classes = 61
num_features = 40


# HYPER PARAMETERS
TRAIN_CAP = 1000
TEST_CAP = 500
NUM_LAYERS = 3
NUM_HIDDEN = 100
LEARNING_RATE = 0.01
NUM_EPOCHS = 40
BATCH_SIZE = 100

SAVE_DIR = "./checkpoint/save"
PLOTTING = True

SAVE_PER_EPOCHS = 1
RESAMPLE_PER_EPOCHS = 20


def initialise_plot():
    plt.ion()
    plt.show()
    plt.gcf().clear()
    plt.title('NH={} NL={} LR={} BS={}'.format(
        NUM_HIDDEN, NUM_LAYERS, LEARNING_RATE, BATCH_SIZE))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


def annotate_max(x, y, ax=None):
    xmax = (x[np.argmax(y)] + 1) * SAVE_PER_EPOCHS
    ymax = y.max()
    text = "Max accuracy\nEpoch={}, Accuracy={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    plt.annotate(text, xy=(0.75, 0.1),
                 xycoords='axes fraction', bbox=bbox_props)


def plot_graph(train_accuracy, test_accuracy):
    plt.gca().set_color_cycle(['red', 'green'])
    plt.axis([0,
              (len(train_accuracy) + 1) * SAVE_PER_EPOCHS,
              min(train_accuracy + test_accuracy) * 0.9,
              max(train_accuracy + test_accuracy) * 1.1])
    plt.plot(np.arange(1, len(train_accuracy) + 1) *
             SAVE_PER_EPOCHS, np.array(train_accuracy))
    plt.plot(np.arange(1, len(test_accuracy) + 1) *
             SAVE_PER_EPOCHS, np.array(test_accuracy))
    x = np.array(np.arange(len(test_accuracy)))
    annotate_max(x, np.asarray(test_accuracy))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.draw()
    plt.pause(0.0001)


def save_plot():
    plt.savefig('./images/{}_{}_{}_{}.png'.format(
        NUM_HIDDEN, NUM_LAYERS, LEARNING_RATE, BATCH_SIZE),
        bbox_inches='tight')


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


def get_mfccs_and_phones(wav_file, sr, trim=False, random_crop=False, length=int(hp.Default.duration / hp.Default.frame_shift + 1)):
    '''
    This is applied in `train1` or `test1` phase.

    args:
    wav_file - wave filename
    sr - sampling ratio
    trim - remove 0th index from mfccs[] and phns[]
    random_crop - retrieve a `length` segment from a random starting point
    length - used with `random_crop`
    '''

    # Load
    wav, sr = librosa.load(wav_file, sr=sr)

    mfccs, _, _ = _get_mfcc_log_spec_and_log_mel_spec(wav, hp.Default.preemphasis, hp.Default.n_fft,
                                                      hp.Default.win_length,
                                                      hp.Default.hop_length)
    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV.wav", "PHN").replace("WAV", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp.Default.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # # Random crop
    # if random_crop:
    #     start = np.random.choice(
    #         range(np.maximum(1, len(mfccs) - length)), 1)[0]
    #     end = start + length
    #     mfccs = mfccs[start:end]
    #     phns = phns[start:end]
    #     assert (len(mfccs) == len(phns))

    # # Padding or crop
    # mfccs = librosa.util.fix_length(mfccs, length, axis=0)
    # phns = librosa.util.fix_length(phns, length, axis=0)
    return mfccs, phns


def load_train_data():
    wav_files = sorted(glob.glob(hp.Train1.data_path))
    x = []
    y = []
    if os.path.isfile(hp.Train1.npz_file_path):
        with np.load(hp.Train1.npz_file_path) as data:
            x = data['mfccs']
            y = data['phns']
    else:
        for i in xrange(len(wav_files)):
            mfccs, phns = get_mfccs_and_phones(wav_files[i], hp.Default.sr)
            x.append(mfccs)
            y.append(phns)
            print("File {}".format(i))
        with open(hp.Train1.npz_file_path, 'wb') as fp:
            np.savez_compressed(fp, mfccs=x, phns=y)

    print("Loaded mfccs and phns from TRAIN data")

    # # Shuffle
    # idx = np.arange(0, len(x))
    # np.random.shuffle(idx)
    # idx = idx[:TRAIN_CAP]
    # x_shuffle = [x[i] for i in idx]
    # y_shuffle = [y[i] for i in idx]
    return np.asarray(x), np.asarray(y)


def load_test_data():
    wav_files = sorted(glob.glob(hp.Test1.data_path))
    x = []
    y = []
    if os.path.isfile(hp.Test1.npz_file_path):
        with np.load(hp.Test1.npz_file_path) as data:
            x = data['mfccs']
            y = data['phns']
    else:
        for i in xrange(len(wav_files)):
            mfccs, phns = get_mfccs_and_phones(wav_files[i], hp.Default.sr)
            x.append(mfccs)
            y.append(phns)
            print("File {}".format(i))
        with open(hp.Test1.npz_file_path, 'wb') as fp:
            np.savez_compressed(fp, mfccs=x, phns=y)

    print("Loaded mfccs and phns from TEST data")

    # # Shuffle
    # idx = np.arange(0, len(x))
    # np.random.shuffle(idx)
    # idx = idx[:TEST_CAP]
    # x_shuffle = [x[i] for i in idx]
    # y_shuffle = [y[i] for i in idx]
    return np.asarray(x), np.asarray(y)


def sample_data(mfccs_array, phns_array):

    length = int(hp.Default.duration / hp.Default.frame_shift + 1)

    for i in range(len(mfccs_array)):
        mfccs = mfccs_array[i]
        phns = phns_array[i]
        # Random crop
        start = np.random.choice(
            range(np.maximum(1, len(mfccs) - length)), 1)[0]
        end = start + length
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

        # Padding or crop
        mfccs = librosa.util.fix_length(mfccs, length, axis=0)
        phns = librosa.util.fix_length(phns, length, axis=0)

        mfccs_array[i], phns_array[i] = mfccs, phns
    return np.asarray(mfccs_array), np.asarray(phns_array)


def get_arguments():
    parser = argparse.ArgumentParser()
    optional = parser.add_argument_group('hyperparams')
    optional.add_argument('--nh', type=int, required=False,
                          help='number of hidden nodes')
    optional.add_argument('--nl', type=int, required=False,
                          help='number of lstm layers')
    optional.add_argument('--epochs', type=int, required=False,
                          help='number of epochs')
    optional.add_argument('--batch_size', type=int,
                          required=False, help='BATCH_SIZE')
    arguments = parser.parse_args()
    global NUM_HIDDEN, NUM_LAYERS, NUM_EPOCHS, BATCH_SIZE
    if arguments.nh:
        NUM_HIDDEN = arguments.nh
    if arguments.nl:
        NUM_LAYERS = arguments.nl
    if arguments.epochs:
        NUM_EPOCHS = arguments.epochs
    if arguments.batch_size:
        BATCH_SIZE = arguments.batch_size
    return arguments


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = np.asarray([one_hot(labels[i]) for i in idx])
    train_seq_len = [len(x) for x in data_shuffle]
    return data_shuffle, labels_shuffle, train_seq_len


def one_hot(indices, depth=num_classes):
    one_hot_labels = np.zeros((len(indices), depth))
    one_hot_labels[np.arange(len(indices)), indices] = 1
    return one_hot_labels


def set_parameters(nh, nl, epochs, batch_size):
    global NUM_HIDDEN, NUM_LAYERS, NUM_EPOCHS, BATCH_SIZE
    NUM_HIDDEN = nh
    NUM_LAYERS = nl
    NUM_EPOCHS = epochs
    BATCH_SIZE = batch_size


def train():

    # Load Train data completely (All 4620 samples, unpadded, uncropped)
    all_train_inputs, all_train_targets = load_train_data()

    # Load Test data completely (All 1680 samples, unpadded, uncropped)
    all_test_inputs, all_test_targets = load_test_data()

    graph = tf.Graph()
    with graph.as_default():
        # Input placeholder of shape [BATCH_SIZE, num_frames, num_mfcc_features]
        inputs = tf.placeholder(tf.float32, [None, None, num_features])

        # Target placeholder of shape [BATCH_SIZE, num_frames, num_phn_classes]
        targets = tf.placeholder(tf.int32, [None, None, num_classes])

        # List of sequence lengths (num_frames)
        seq_len = tf.placeholder(tf.int32, [None])

        # Get a basic LSTM cell with dropout for use in RNN
        def get_a_cell(lstm_size, keep_prob=1.0):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(
                lstm, output_keep_prob=keep_prob)
            return drop

        # Make a multi layer RNN of NUM_LAYERS layers of cells
        stack = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(NUM_HIDDEN) for _ in range(NUM_LAYERS)])

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
            # initialise the variables
            sess.run(init_op)
            print("Model initialised.\n")

        train_accuracy = []
        test_accuracy = []
        if PLOTTING:
            initialise_plot()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_cost = 0
            start = time.time()

            if (epoch % RESAMPLE_PER_EPOCHS == 0 or epoch == 1):
                train_inputs, train_targets = sample_data(
                    all_train_inputs, all_train_targets)
                train_targets = np.array(list(train_targets))
                train_inputs = np.array(list(train_inputs))

                train_targets = train_targets.astype(int)
                train_inputs = (train_inputs - np.mean(train_inputs)) / \
                    np.std(train_inputs)

                num_examples = len(train_targets)

                test_inputs, test_targets = sample_data(
                    all_test_inputs, all_test_targets)

                test_targets = np.array(list(test_targets))
                test_inputs = np.array(list(test_inputs))

                test_targets = test_targets.astype(int)
                test_inputs = (test_inputs - np.mean(test_inputs)) / \
                    np.std(test_inputs)
                print("Re-sampled data (2sec of every wav)")

            for batch in range(int(num_examples / BATCH_SIZE)):

                batch_x, batch_y, batch_seq_len = next_batch(
                    BATCH_SIZE, train_inputs, train_targets)

                feed = {inputs: batch_x,
                        targets: batch_y,
                        seq_len: batch_seq_len}

                batch_cost, _ = sess.run([cross_entropy, optimizer], feed)
                train_cost += batch_cost * BATCH_SIZE

            train_cost /= num_examples
            print("Epoch {}/{}, train_cost = {:.3f}, time = {:.3f}".format(
                epoch, NUM_EPOCHS, train_cost, time.time() - start))

            if (epoch % SAVE_PER_EPOCHS == 0):
                save_path = saver.save(sess, SAVE_PATH)
                print("Model saved in path: %s" % save_path)

                batch_x, batch_y, batch_seq_len = next_batch(
                    TRAIN_CAP, train_inputs, train_targets)

                train_acc = sess.run(accuracy, feed_dict={
                    inputs: batch_x,
                    targets: batch_y,
                    seq_len: batch_seq_len})

                batch_x, batch_y, batch_seq_len = next_batch(
                    TEST_CAP, test_inputs, test_targets)

                test_acc = sess.run(accuracy, feed_dict={
                    inputs: batch_x,
                    targets: batch_y,
                    seq_len: batch_seq_len})

                log = "\nEpoch {}/{}, train_cost = {:.3f}, " + \
                    "train_acc = {:.3f}, test_acc = {:.3f} time = {:.3f}\n"
                print(log.format(epoch, NUM_EPOCHS, train_cost, train_acc,
                                 test_acc, time.time() - start))

                train_accuracy.append(train_acc)
                test_accuracy.append(test_acc)

                if PLOTTING:
                    plot_graph(train_accuracy, test_accuracy)

        if PLOTTING:
            save_plot()


if __name__ == '__main__':
    args = get_arguments()
    params_arr = [{'nh': 100, 'nl': 2, 'epochs': 50, 'batch_size': 100},
                  {'nh': 75, 'nl': 3, 'epochs': 50, 'batch_size': 100},
                  {'nh': 100, 'nl': 3, 'epochs': 50, 'batch_size': 100},
                  {'nh': 75, 'nl': 4, 'epochs': 50, 'batch_size': 100}]
    for params in params_arr:
        set_parameters(**params)
        train()
