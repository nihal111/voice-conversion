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
NUM_LAYERS = 2
NUM_HIDDEN = 100
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 100
KEEP_PROB = 0.9

SAVE_DIR = "./checkpoint2/save"
PLOTTING = True

SAVE_PER_EPOCHS = 1
RESAMPLE_PER_EPOCHS = 10


def db_to_amplitude(x):
    return 10.0**(x / 10.0)


def preemphasis(x, coeff=0.97):
    '''
    Applies a pre-emphasis filter on x
    '''
    return signal.lfilter([1, -coeff], [1], x)


def deemphasis(x, coeff=0.97):
    return signal.lfilter([1], [1, -coeff], x)


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
    phn_file = wav_file.replace("wav", "lab")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        if(line != "#"):
            start_time, _, phn = line.split()
            bnd = int(float(start_time) * sr // hp.Default.hop_length)
            phns[bnd:] = phn2idx[phn]
            bnd_list.append(bnd)

    # Replace pau with h# for consistency with TIMIT
    phns[phns == 44.] = 0.

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


def load_test_data(phn_file):
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(1000,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        # For TIMIT files
        # start_point, end_point, phn = line.split()
        # bnd = int(start_point) // hp.Default.hop_length
        # phns[bnd:] = phn2idx[phn]
        # bnd_list.append(bnd)
        # For Arctic files
        bnd_list.append(0)
        prev_bnd = 0
        if(line != "#"):
            end_time, _, phn = line.split()
            bnd = int(float(end_time) * sr // hp.Default.hop_length)
            phns[prev_bnd:bnd] = phn2idx[phn]
            bnd_list.append(bnd)
            prev_bnd = bnd
    phns[phns == 44.] = 0.
    end_point = bnd_list[-1]
    phns = phns[:(int(end_point) // hp.Default.hop_length)]
    start, end = bnd_list[1], bnd_list[-1]
    phns = phns[start:end]
    # print (phns)
    return np.array([phns])


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('predict_file', type=str, help='predict file path')
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


def one_hot(indices, depth=num_classes):
    one_hot_labels = np.zeros((len(indices), depth))
    one_hot_labels[np.arange(len(indices)), indices] = 1
    return one_hot_labels


def set_parameters(nh, nl, epochs, batch_size, keep_prob):
    global NUM_HIDDEN, NUM_LAYERS, NUM_EPOCHS, BATCH_SIZE, KEEP_PROB
    NUM_HIDDEN = nh
    NUM_LAYERS = nl
    NUM_EPOCHS = epochs
    BATCH_SIZE = batch_size
    KEEP_PROB = keep_prob


def spectrogram2wav(mag, n_fft, win_length, hop_length, num_iters, phase_angle=None, length=None):
    assert(num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = mag * np.exp(1.j * phase_angle)
    for i in range(num_iters):
        wav = librosa.istft(spec, win_length=win_length,
                            hop_length=hop_length, length=length)
        if i != num_iters - 1:
            spec = librosa.stft(
                wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            _, phase = librosa.magphase(spec)
            phase_angle = np.angle(phase)
            spec = mag * np.exp(1.j * phase_angle)
    return deemphasis(wav)


def _get_wav_from_mfccs(mfccs, preemphasis_coeff, n_fft, win_length, hop_length, n_wav):
    dctm = librosa.filters.dct(hp.Default.n_mfcc, hp.Default.n_mels)
    mel_basis = librosa.filters.mel(
        hp.Default.sr, hp.Default.n_fft, hp.Default.n_mels)
    bin_scaling = 1.0 / \
        np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    mel_db = np.dot(dctm.T, mfccs.T)
    mel = db_to_amplitude(mel_db)
    recon_magsq = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, mel)
    mag = np.sqrt(recon_magsq)
    #excitation = np.random.randn(n_wav)
    #E = librosa.stft(excitation, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    #recon = librosa.core.istft(np.sqrt(recon_stft), hop_length=hop_length, win_length=win_length)
    recon = spectrogram2wav(mag, n_fft, win_length,
                            hop_length, hp.Default.n_iter)
    recon = deemphasis(recon, coeff=preemphasis_coeff)
    return recon


if __name__ == '__main__':
    args = get_arguments()
    predict_file = args.predict_file

    graph = tf.Graph()
    with graph.as_default():
        # Input placeholder of shape [BATCH_SIZE, num_frames, num_phn_classes]
        inputs = tf.placeholder(tf.float32, [None, None, num_classes])

        # Target placeholder of shape [BATCH_SIZE, num_frames, num__mfcc_features]
        targets = tf.placeholder(tf.int32, [None, None, num_features])

        # List of sequence lengths (num_frames)
        seq_len = tf.placeholder(tf.int32, [None])

        keep_prob = tf.placeholder(tf.float32, shape=())

        mean = tf.Variable(-3.643601, dtype=tf.float32)

        std_dev = tf.Variable(2.283052, dtype=tf.float32)

        # Get a GRU cell with dropout for use in RNN
        def get_a_cell(gru_size, keep_prob=1.0):
            gru = tf.nn.rnn_cell.GRUCell(gru_size)
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
                                             num_features],
                                            stddev=0.1))
        # Zero initialization
        b = tf.Variable(tf.constant(0., shape=[num_features]))

        # Doing the affine projection
        predictions = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        predictions = tf.reshape(predictions, [batch_s, -1, num_features])

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
        SAVE_PATH = SAVE_DIR + '_bigru_{}_{}_{}_{}_{}/model.ckpt'.format(
            NUM_HIDDEN, NUM_LAYERS, LEARNING_RATE, BATCH_SIZE, KEEP_PROB)
        try:
            saver.restore(sess, SAVE_PATH)
            print("Model restored.\n")
        except:
            # initialise the variables
            sess.run(init_op)
            print("Model initialised.\n")

        predict_inputs = load_test_data(predict_file)
        predict_inputs = np.array(predict_inputs).astype(int)

        predict_inputs = np.asarray([one_hot(x) for x in predict_inputs])

        num_examples = len(predict_inputs)
        predict_seq_len = [len(x) for x in predict_inputs]

        feed = {inputs: predict_inputs,
                seq_len: predict_seq_len,
                keep_prob: 1.0}

        mfccs = sess.run(scaled_predictions, feed)[0]

        print(mfccs)
        print(mfccs.shape)

        audio = _get_wav_from_mfccs(mfccs,
                                    hp.Default.preemphasis,
                                    hp.Default.n_fft,
                                    hp.Default.win_length,
                                    hp.Default.hop_length,
                                    (len(mfccs) - 1) * hp.Default.hop_length)
        librosa.output.write_wav(
            "SA1_pred.wav", audio, hp.Default.sr, norm=True)
