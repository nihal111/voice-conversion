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

# HYPER PARAMETERS
TRAIN_CAP = TEST_CAP = 50
LAYERS = [140, 200]
NUM_HIDDEN = 200
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 5
KEEP_PROB = 1.0

SAVE_DIR = "./checkpoint2/save"
PLOTTING = True

SAVE_PER_EPOCHS = 1
RESAMPLE_PER_EPOCHS = 10


def initialise_plot():
    plt.ion()
    plt.show()
    plt.gcf().clear()
    plt.title('NH={} L={} LR={} BS={} KP={}'.format(
        NUM_HIDDEN, LAYERS, LEARNING_RATE, BATCH_SIZE, KEEP_PROB))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')


def annotate_min(x, y, ax=None):
    xmin = (x[np.argmin(y)] + 1) * SAVE_PER_EPOCHS
    ymin = y.min()
    text = "Min Error\nEpoch={}\nMSE={:.3f}".format(xmin, ymin)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    plt.annotate(text, xy=(0.75, 0.8),
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
    plt.savefig('./images2/bigru_{}_{}_{}_{}.png'.format(
        NUM_HIDDEN, LEARNING_RATE, BATCH_SIZE, KEEP_PROB),
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


def get_mags_and_phones(wav_file, sr, trim=False, random_crop=False, length=int(hp.Default.duration / hp.Default.frame_shift + 1)):
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

    _, mags, _ = _get_mfcc_log_spec_and_log_mel_spec(wav, hp.Default.preemphasis, hp.Default.n_fft,
                                                     hp.Default.win_length,
                                                     hp.Default.hop_length)
    # timesteps
    num_timesteps = mags.shape[0]

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
        phns = phns[start:end]
        assert (len(mags) == len(phns))

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
    return mags, phns


def load_train_data():
    wav_files = sorted(glob.glob(hp.Train2.data_path))
    x = []
    y = []
    if os.path.isfile(hp.Train2.mag_npz_file_path):
        with np.load(hp.Train2.mag_npz_file_path) as data:
            x = data['mags']
            y = data['phns']
    else:
        for i in xrange(len(wav_files)):
            mags, phns = get_mags_and_phones(wav_files[i], hp.Default.sr)
            x.append(mags)
            y.append(phns)
            print("File {}".format(i))
        with open(hp.Train2.mag_npz_file_path, 'wb') as fp:
            np.savez_compressed(fp, mags=x, phns=y)

    print("Loaded mags and phns from TRAIN data")

    # # Shuffle
    # idx = np.arange(0, len(x))
    # np.random.shuffle(idx)
    # idx = idx[:TRAIN_CAP]
    # x_shuffle = [x[i] for i in idx]
    # y_shuffle = [y[i] for i in idx]
    return np.asarray(x), np.asarray(y)


def load_test_data():
    wav_files = sorted(glob.glob(hp.Test2.data_path))
    x = []
    y = []
    if os.path.isfile(hp.Test2.mag_npz_file_path):
        with np.load(hp.Test2.mag_npz_file_path) as data:
            x = data['mags']
            y = data['phns']
    else:
        for i in xrange(len(wav_files)):
            mags, phns = get_mags_and_phones(wav_files[i], hp.Default.sr)
            x.append(mags)
            y.append(phns)
            print("File {}".format(i))
        with open(hp.Test2.mag_npz_file_path, 'wb') as fp:
            np.savez_compressed(fp, mags=x, phns=y)

    print("Loaded mags and phns from TEST data")

    # # Shuffle
    # idx = np.arange(0, len(x))
    # np.random.shuffle(idx)
    # idx = idx[:TEST_CAP]
    # x_shuffle = [x[i] for i in idx]
    # y_shuffle = [y[i] for i in idx]
    return np.asarray(x), np.asarray(y)


def sample_data(mags_array, phns_array):

    length = int(hp.Default.duration / hp.Default.frame_shift + 1)

    for i in range(len(mags_array)):
        mags = mags_array[i]
        phns = phns_array[i]
        # Random crop
        start = np.random.choice(
            range(np.maximum(1, len(mags) - length)), 1)[0]
        end = start + length
        mags = mags[start:end]
        phns = phns[start:end]
        assert (len(mags) == len(phns))

        # Padding or crop
        mags = librosa.util.fix_length(mags, length, axis=0)
        phns = librosa.util.fix_length(phns, length, axis=0)

        mags_array[i], phns_array[i] = mags, phns
    return np.asarray(mags_array), np.asarray(phns_array)


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
    global NUM_EPOCHS, BATCH_SIZE
    if arguments.epochs:
        NUM_EPOCHS = arguments.epochs
    if arguments.batch_size:
        BATCH_SIZE = arguments.batch_size
    return arguments


def next_batch(num, inputs, outputs):
    '''
    Return a total of `num` random samples and labels.
    inputs is phns and outputs is mfccs
    '''
    idx = np.arange(0, len(inputs))
    np.random.shuffle(idx)
    idx = idx[:num]
    outputs_shuffle = [outputs[i] for i in idx]
    inputs_shuffle = np.asarray([one_hot(inputs[i]) for i in idx])
    train_seq_len = [len(x) for x in inputs_shuffle]
    return inputs_shuffle, outputs_shuffle, train_seq_len


def one_hot(indices, depth=num_classes):
    one_hot_labels = np.zeros((len(indices), depth))
    one_hot_labels[np.arange(len(indices)), indices] = 1
    return one_hot_labels


def set_parameters(nl, epochs, batch_size, keep_prob):
    global NUM_HIDDEN, LAYERS, NUM_EPOCHS, BATCH_SIZE, KEEP_PROB
    NUM_HIDDEN = nl[-1]
    LAYERS = nl
    NUM_EPOCHS = epochs
    BATCH_SIZE = batch_size
    KEEP_PROB = keep_prob


def train():

    # Load Train data completely (All 4620 samples, unpadded, uncropped)
    all_train_targets, all_train_inputs = load_train_data()

    train_mean = np.mean(np.concatenate(all_train_targets).ravel())
    train_std_dev = np.std(np.concatenate(all_train_targets).ravel())

    # Load Test data completely (All 1680 samples, unpadded, uncropped)
    all_test_targets, all_test_inputs = load_test_data()

    graph = tf.Graph()
    with graph.as_default():
        # Input placeholder of shape [BATCH_SIZE, num_frames, num_phn_classes]
        inputs = tf.placeholder(tf.float32, [None, None, num_classes])

        # Target placeholder of shape [BATCH_SIZE, num_frames, num__mfcc_features]
        targets = tf.placeholder(tf.int32, [None, None, num_mags])

        # List of sequence lengths (num_frames)
        seq_len = tf.placeholder(tf.int32, [None])

        keep_prob = tf.placeholder(tf.float32, shape=())

        mean = tf.Variable(train_mean)

        std_dev = tf.Variable(train_std_dev)

        # Get a GRU cell with dropout for use in RNN
        def get_a_cell(gru_size, keep_prob=1.0):
            gru = tf.nn.rnn_cell.GRUCell(gru_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(
                gru, output_keep_prob=keep_prob)
            return drop

        # Make a multi layer RNN of NUM_LAYERS layers of cells
        stack = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(num_hidden, keep_prob) for num_hidden in LAYERS])

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
        W = tf.Variable(tf.truncated_normal([2 * NUM_HIDDEN,
                                             num_mags],
                                            stddev=0.1))
        # Zero initialization
        b = tf.Variable(tf.constant(0., shape=[num_mags]))

        # Doing the affine projection
        predictions = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        predictions = tf.reshape(predictions, [batch_s, -1, num_mags])

        scaled_predictions = predictions * std_dev + mean

        # mse_loss = tf.reduce_mean(
        #     tf.losses.mean_squared_error(
        #         predictions=predictions, labels=targets))
        # define an accuracy assessment operation
        mse_loss = tf.losses.mean_squared_error(predictions, targets)

        optimizer = tf.train.AdamOptimizer(
            LEARNING_RATE).minimize(mse_loss)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        SAVE_PATH = SAVE_DIR + '_mag_bigru_{}_{}_{}_{}/model.ckpt'.format(
            NUM_HIDDEN, LEARNING_RATE, BATCH_SIZE, KEEP_PROB)
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
                train_targets, train_inputs = sample_data(
                    all_train_targets, all_train_inputs)
                train_targets = np.array(list(train_targets))
                train_inputs = np.array(list(train_inputs))

                train_inputs = train_inputs.astype(int)

                train_targets = (train_targets - train_mean) / train_std_dev

                num_examples = len(train_inputs)

                test_targets, test_inputs = sample_data(
                    all_test_targets, all_test_inputs)

                test_targets = np.array(list(test_targets))
                test_inputs = np.array(list(test_inputs))

                test_inputs = test_inputs.astype(int)
                test_targets = (test_targets - train_mean) / train_std_dev
                print("Re-sampled data (2sec of every wav)")

            for batch in range(int(num_examples / BATCH_SIZE)):

                batch_x, batch_y, batch_seq_len = next_batch(
                    BATCH_SIZE, train_inputs, train_targets)

                feed = {inputs: batch_x,
                        targets: batch_y,
                        seq_len: batch_seq_len,
                        keep_prob: KEEP_PROB}

                batch_cost, _ = sess.run([mse_loss, optimizer], feed)
                train_cost += batch_cost * BATCH_SIZE

            train_cost /= num_examples
            print("Epoch {}/{}, train_cost = {:.3f}, time = {:.3f}".format(
                epoch, NUM_EPOCHS, train_cost, time.time() - start))

            if (epoch % SAVE_PER_EPOCHS == 0):
                save_path = saver.save(sess, SAVE_PATH)
                print("Model saved in path: %s" % save_path)

                batch_x, batch_y, batch_seq_len = next_batch(
                    TRAIN_CAP, train_inputs, train_targets)

                train_err = sess.run(mse_loss, feed_dict={
                    inputs: batch_x,
                    targets: batch_y,
                    seq_len: batch_seq_len,
                    keep_prob: 1.0})

                batch_x, batch_y, batch_seq_len = next_batch(
                    TEST_CAP, test_inputs, test_targets)

                test_err = sess.run(mse_loss, feed_dict={
                    inputs: batch_x,
                    targets: batch_y,
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
        {'nl': [61, 124, 257], 'epochs': 30, 'batch_size': 25, 'keep_prob': 0.6}
    ]
    for params in params_arr:
        set_parameters(**params)
        train()
