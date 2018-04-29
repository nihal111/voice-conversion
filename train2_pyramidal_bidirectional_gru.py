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
num_mags = hp.Default.n_fft / 2 + 1
num_mels = 80


# HYPER PARAMETERS
LAYERS1 = [65, 75]
LAYERS2 = [140, 200]
NUM_HIDDEN1 = 75
NUM_HIDDEN2 = 200
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 100
KEEP_PROB = 0.9
TRAIN_CAP = TEST_CAP = 100

SAVE_DIR = "./checkpoint2/savepyramidal"
PLOTTING = True

SAVE_PER_EPOCHS = 1
RESAMPLE_PER_EPOCHS = 10


def initialise_plot():
    plt.ion()
    plt.show()
    plt.gcf().clear()
    plt.title('L1={} L2={} LR={} BS={} KP={}'.format(
        LAYERS1, LAYERS2, LEARNING_RATE, BATCH_SIZE, KEEP_PROB))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')


def annotate_min(x, y, ax=None):
    xmin = (x[np.argmin(y)] + 1) * SAVE_PER_EPOCHS
    ymin = y.min()
    text = "Min Error\nEpoch={}\nMSE={:.3f}".format(xmin, ymin)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    plt.annotate(text, xy=(0.80, 0.85),
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
    annotate_min(x, np.asarray(test_accuracy))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.draw()
    plt.pause(0.0001)


def save_plot():
    plt.savefig('./images2/bigru_{}_{}_{}_{}_{}.png'.format(
        LAYERS1, LAYERS2, LEARNING_RATE, BATCH_SIZE, KEEP_PROB),
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


def get_mfcc_log_spec_and_log_mel_spec(wav, preemphasis_coeff,
                                       n_fft, win_length, hop_length):
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


def get_mags_mels_and_phones(wav_file, sr, trim=False, random_crop=False, length=int(hp.Default.duration / hp.Default.frame_shift + 1)):
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

    _, mags, mels = get_mfcc_log_spec_and_log_mel_spec(wav,
                                                       hp.Default.preemphasis,
                                                       hp.Default.n_fft,
                                                       hp.Default.win_length,
                                                       hp.Default.hop_length)
    # timesteps
    num_timesteps = mels.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("wav", "lab")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    bnd_list.append(0)
    prev_bnd = 0
    for line in open(phn_file, 'r').read().splitlines():
        if(line != "#"):
            end_time, _, phn = line.split()
            bnd = int(float(end_time) * sr // hp.Default.hop_length)
            phns[prev_bnd:bnd] = phn2idx[phn]
            bnd_list.append(bnd)
            prev_bnd = bnd

    # Replace pau with h# for consistency with TIMIT
    phns[phns == 44.] = 0.

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mags = mags[start:end]
        mels = mels[start:end]
        phns = phns[start:end]
        assert (len(mags) == len(phns))
        assert (len(mels) == len(phns))

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
    return mags, mels, phns


def load_train_data():
    wav_files = sorted(glob.glob(hp.Train2.data_path))
    x = []
    y = []
    z = []
    if os.path.isfile(hp.Train2.npz_file_path):
        with np.load(hp.Train2.npz_file_path) as data:
            x = data['mags']
            y = data['mels']
            z = data['phns']
    else:
        for i in xrange(len(wav_files)):
            mags, mels, phns = get_mags_mels_and_phones(
                wav_files[i], hp.Default.sr)
            x.append(mags)
            y.append(mels)
            z.append(phns)
            print("File {}".format(i))
        with open(hp.Train2.npz_file_path, 'wb') as fp:
            np.savez_compressed(fp, mags=x, mels=y, phns=z)

    print("Loaded mfccs and phns from TRAIN data")

    # # Shuffle
    # idx = np.arange(0, len(x))
    # np.random.shuffle(idx)
    # idx = idx[:TRAIN_CAP]
    # x_shuffle = [x[i] for i in idx]
    # y_shuffle = [y[i] for i in idx]
    return np.asarray(x), np.asarray(y), np.asarray(z)


def load_test_data():
    wav_files = sorted(glob.glob(hp.Test2.data_path))
    x = []
    y = []
    z = []
    if os.path.isfile(hp.Test2.npz_file_path):
        with np.load(hp.Test2.npz_file_path) as data:
            x = data['mags']
            y = data['mels']
            z = data['phns']
    else:
        for i in xrange(len(wav_files)):
            mags, mels, phns = get_mags_mels_and_phones(
                wav_files[i], hp.Default.sr)
            x.append(mags)
            y.append(mels)
            z.append(phns)
            print("File {}".format(i))
        with open(hp.Test2.npz_file_path, 'wb') as fp:
            np.savez_compressed(fp, mags=x, mels=y, phns=z)

    print("Loaded mfccs and phns from TEST data")

    # # Shuffle
    # idx = np.arange(0, len(x))
    # np.random.shuffle(idx)
    # idx = idx[:TEST_CAP]
    # x_shuffle = [x[i] for i in idx]
    # y_shuffle = [y[i] for i in idx]
    return np.asarray(x), np.asarray(y), np.asarray(z)


def sample_data(mags_array, mels_array, phns_array):

    length = int(hp.Default.duration / hp.Default.frame_shift + 1)

    for i in range(len(mags_array)):
        mags = mags_array[i]
        mels = mels_array[i]
        phns = phns_array[i]
        # Random crop
        start = np.random.choice(
            range(np.maximum(1, len(mags) - length)), 1)[0]
        end = start + length
        mags = mags[start:end]
        mels = mels[start:end]
        phns = phns[start:end]
        assert (len(mags) == len(phns))

        # Padding or crop
        mags = librosa.util.fix_length(mags, length, axis=0)
        mels = librosa.util.fix_length(mels, length, axis=0)
        phns = librosa.util.fix_length(phns, length, axis=0)

        mags_array[i], phns_array[i] = mags, phns
        mels_array[i], phns_array[i] = mels, phns
    return np.asarray(mags_array), np.asarray(mels_array), \
        np.asarray(phns_array)


def get_arguments():
    parser = argparse.ArgumentParser()
    optional = parser.add_argument_group('hyperparams')
    # optional.add_argument('--nh', type=int, required=False,
    #                       help='number of hidden nodes')
    # optional.add_argument('--nl', type=int, required=False,
    #                       help='number of lstm layers')
    optional.add_argument('--epochs', type=int, required=False,
                          help='number of epochs')
    optional.add_argument('--batch_size', type=int,
                          required=False, help='BATCH_SIZE')
    arguments = parser.parse_args()
    global NUM_EPOCHS, BATCH_SIZE
    if arguments.epochs:
        NUM_EPOCHS = arguments.epochs
    if arguments.batch_size:
        BATCH_SIZE = arguments.batch_size
    return arguments


def next_batch(num, inputs, outputs1, outputs2):
    '''
    Return a total of `num` random samples and labels.
    inputs is phns and outputs is mfccs
    '''
    idx = np.arange(0, len(inputs))
    np.random.shuffle(idx)
    idx = idx[:num]
    outputs1_shuffle = [outputs1[i] for i in idx]
    outputs2_shuffle = [outputs2[i] for i in idx]
    inputs_shuffle = np.asarray([one_hot(inputs[i]) for i in idx])
    train_seq_len = [len(x) for x in inputs_shuffle]
    return inputs_shuffle, outputs1_shuffle, outputs2_shuffle, train_seq_len


def one_hot(indices, depth=num_classes):
    one_hot_labels = np.zeros((len(indices), depth))
    one_hot_labels[np.arange(len(indices)), indices] = 1
    return one_hot_labels


def set_parameters(layers1, layers2, epochs, batch_size, keep_prob):
    global NUM_HIDDEN1, NUM_HIDDEN2, LAYERS1, LAYERS2, NUM_EPOCHS, BATCH_SIZE, KEEP_PROB
    NUM_HIDDEN1 = layers1[-1]
    LAYERS1 = layers1
    NUM_HIDDEN2 = layers2[-1]
    LAYERS2 = layers2
    NUM_EPOCHS = epochs
    BATCH_SIZE = batch_size
    KEEP_PROB = keep_prob


def train():

    # Load Train data completely (All 4620 samples, unpadded, uncropped)
    all_train_mags, all_train_mels, all_train_inputs = load_train_data()

    train_mags_mean = np.mean(np.concatenate(all_train_mags).ravel())
    train_mags_std_dev = np.std(np.concatenate(all_train_mags).ravel())
    train_mels_mean = np.mean(np.concatenate(all_train_mels).ravel())
    train_mels_std_dev = np.std(np.concatenate(all_train_mels).ravel())

    print(train_mags_mean)
    print(train_mags_std_dev)
    print(train_mels_mean)
    print(train_mels_std_dev)

    # Load Test data completely (All 1680 samples, unpadded, uncropped)
    all_test_mags, all_test_mels, all_test_inputs = load_test_data()

    graph = tf.Graph()
    with graph.as_default():
        # Input placeholder of shape [BATCH_SIZE, num_frames, num_phn_classes]
        inputs = tf.placeholder(tf.float32, [None, None, num_classes])

        # Target placeholder of shape [BATCH_SIZE, num_frames, num__mels]
        target_mels = tf.placeholder(tf.int32, [None, None, num_mels])

        # Target placeholder of shape [BATCH_SIZE, num_frames, num__mags]
        target_mags = tf.placeholder(tf.int32, [None, None, num_mags])

        # List of sequence lengths (num_frames)
        seq_len = tf.placeholder(tf.int32, [None])

        keep_prob = tf.placeholder(tf.float32, shape=())

        mags_mean = tf.Variable(train_mags_mean, dtype=tf.float32)
        mags_std_dev = tf.Variable(train_mags_std_dev, dtype=tf.float32)
        mels_mean = tf.Variable(train_mels_mean, dtype=tf.float32)
        mels_std_dev = tf.Variable(train_mels_std_dev, dtype=tf.float32)

        # Get a GRU cell with dropout for use in RNN
        def get_a_cell(gru_size, keep_prob=1.0):
            gru = tf.nn.rnn_cell.GRUCell(gru_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(
                gru, output_keep_prob=keep_prob)
            return drop

        # Make a multi layer RNN of LAYERS layers of cells
        stack1_fw = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(num_hidden, keep_prob) for num_hidden in LAYERS1])

        stack1_bw = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(num_hidden, keep_prob) for num_hidden in LAYERS1])

        (mel_output_fw, mel_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            stack1_fw, stack1_bw, inputs, seq_len, dtype=tf.float32)
        mel_outputs = tf.concat([mel_output_fw, mel_output_bw], axis=2)

        # Save input shape for restoring later
        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        # mel_outputs is now of shape [BATCH_SIZE*num_frames, NUM_HIDDEN]
        # So the same weights are trained for each timestep of each sequence
        mel_outputs = tf.reshape(mel_outputs, [-1, 2 * NUM_HIDDEN1])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        W1 = tf.Variable(tf.truncated_normal([2 * NUM_HIDDEN1,
                                              num_mels],
                                             stddev=0.1))
        # Zero initialization
        b1 = tf.Variable(tf.constant(0., shape=[num_mels]))

        # Doing the affine projection
        mels_predictions = tf.matmul(mel_outputs, W1) + b1

        # Reshaping back to the original shape
        mels_predictions = tf.reshape(
            mels_predictions, [batch_s, -1, num_mels])

        scaled_mels_predictions = mels_predictions * mels_std_dev + mels_mean

        stack2_fw = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(num_hidden, keep_prob) for num_hidden in LAYERS2])

        stack2_bw = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(num_hidden, keep_prob) for num_hidden in LAYERS2])

        (mag_output_fw, mag_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            stack2_fw, stack2_bw, mels_predictions, seq_len,
            dtype=tf.float32, scope="bi_RNN2")
        mag_outputs = tf.concat([mag_output_fw, mag_output_bw], axis=2)

        mag_outputs = tf.reshape(mag_outputs, [-1, 2 * NUM_HIDDEN2])

        W2 = tf.Variable(tf.truncated_normal([2 * NUM_HIDDEN2,
                                              num_mags],
                                             stddev=0.1))
        # Zero initialization
        b2 = tf.Variable(tf.constant(0., shape=[num_mags]))

        # Doing the affine projection
        mags_predictions = tf.matmul(mag_outputs, W2) + b2

        # Reshaping back to the original shape
        mags_predictions = tf.reshape(
            mags_predictions, [batch_s, -1, num_mags])

        scaled_mags_predictions = mags_predictions * mags_std_dev + mags_mean

        mels_mse_loss = tf.losses.mean_squared_error(
            mels_predictions, target_mels)
        mags_mse_loss = tf.losses.mean_squared_error(
            mags_predictions, target_mags)

        total_mse_loss = mels_mse_loss + mags_mse_loss

        optimizer = tf.train.AdamOptimizer(
            LEARNING_RATE).minimize(total_mse_loss)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        SAVE_PATH = SAVE_DIR + '_bigru_{}_{}_{}_{}_{}/model.ckpt'.format(
            LAYERS1, LAYERS2, LEARNING_RATE, BATCH_SIZE, KEEP_PROB)
        try:
            saver.restore(sess, SAVE_PATH)
            print("Model restored.\n")
        except:
            # initialise the variables
            sess.run(init_op)
            print("Model initialised.\n")

        train_errors = []
        test_errors = []
        if PLOTTING:
            initialise_plot()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_cost = 0
            start = time.time()

            if (epoch % RESAMPLE_PER_EPOCHS == 0 or epoch == 1):
                # sample_data returns mfccs,phns
                train_mags, train_mels, train_inputs = sample_data(
                    all_train_mags, all_train_mels, all_train_inputs)
                train_mags = np.array(list(train_mags))
                train_mels = np.array(list(train_mels))
                train_inputs = np.array(list(train_inputs))

                train_inputs = train_inputs.astype(int)
                train_mags = (train_mags - train_mags_mean) / \
                    train_mags_std_dev
                train_mels = (train_mels - train_mels_mean) / \
                    train_mels_std_dev

                num_examples = len(train_inputs)

                test_mags, test_mels, test_inputs = sample_data(
                    all_test_mags, all_test_mels, all_test_inputs)

                test_mags = np.array(list(test_mags))
                test_mels = np.array(list(test_mels))
                test_inputs = np.array(list(test_inputs))

                test_inputs = test_inputs.astype(int)
                test_mags = (test_mags - train_mags_mean) / \
                    train_mags_std_dev
                test_mels = (test_mels - train_mels_mean) / \
                    train_mels_std_dev
                print("Re-sampled data (2sec of every wav)")

            for batch in range(int(num_examples / BATCH_SIZE)):

                batch_x, batch_y_mags, batch_y_mels, \
                    batch_seq_len = next_batch(
                        BATCH_SIZE, train_inputs, train_mags, train_mels)

                feed = {inputs: batch_x,
                        target_mels: batch_y_mels,
                        target_mags: batch_y_mags,
                        seq_len: batch_seq_len,
                        keep_prob: KEEP_PROB}

                batch_cost, _ = sess.run([total_mse_loss, optimizer], feed)
                train_cost += batch_cost * BATCH_SIZE

            train_cost /= num_examples
            print("Epoch {}/{}, train_cost = {:.3f}, time = {:.3f}".format(
                epoch, NUM_EPOCHS, train_cost, time.time() - start))

            if (epoch % SAVE_PER_EPOCHS == 0):
                save_path = saver.save(sess, SAVE_PATH)
                print("Model saved in path: %s" % save_path)

                batch_x, batch_y_mags, batch_y_mels, \
                    batch_seq_len = next_batch(
                        TRAIN_CAP, train_inputs,
                        train_mags, train_mels)

                train_err = sess.run(total_mse_loss, feed_dict={
                    inputs: batch_x,
                    target_mels: batch_y_mels,
                    target_mags: batch_y_mags,
                    seq_len: batch_seq_len,
                    keep_prob: 1.0})

                batch_x, batch_y_mags, batch_y_mels, \
                    batch_seq_len = next_batch(
                        TEST_CAP, test_inputs,
                        test_mags, test_mels)

                test_err = sess.run(total_mse_loss, feed_dict={
                    inputs: batch_x,
                    target_mels: batch_y_mels,
                    target_mags: batch_y_mags,
                    seq_len: batch_seq_len,
                    keep_prob: 1.0})

                log = "\nEpoch {}/{}, train_error = {:.3f}, " + \
                    "test_error = {:.3f} time = {:.3f}\n"
                print(log.format(epoch, NUM_EPOCHS, train_err,
                                 test_err, time.time() - start))

                train_errors.append(train_err)
                test_errors.append(test_err)

                if PLOTTING:
                    plot_graph(train_errors, test_errors)

        if PLOTTING:
            save_plot()


if __name__ == '__main__':
    args = get_arguments()
    params_arr = [
        {'layers1': [65, 75], 'layers2':[140, 200], 'epochs': 50,
            'batch_size': 20, 'keep_prob': 0.8}
    ]
    for params in params_arr:
        set_parameters(**params)
        train()
