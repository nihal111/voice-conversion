import librosa
import numpy as np
from IPython.lib.display import Audio
import hyperparams as hp
from scipy import signal
import sys


def db_to_amplitude(x):
    return 10.0**(x / 10.0)


def preemphasis(x, coeff=0.97):
    '''
    Applies a pre-emphasis filter on x
    '''
    return signal.lfilter([1, -coeff], [1], x)


def deemphasis(x, coeff=0.97):
    return signal.lfilter([1], [1, -coeff], x)


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


def spectrogram2wav(mag, n_fft, win_length, hop_length, num_iters, phase_angle=None, length=None):
    assert(num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = mag * np.exp(1.j * phase_angle)
    print("\nspec:" + str(spec.shape))
    for i in range(num_iters):
        wav = librosa.istft(spec, win_length=win_length,
                            hop_length=hop_length, length=length)
        print("wav:" + str(wav.shape))
        if i != num_iters - 1:
            spec = librosa.stft(
                wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            print("spec:" + str(spec.shape))
            _, phase = librosa.magphase(spec)
            print("phase:" + str(phase.shape))
            phase_angle = np.angle(phase)
            print("phase_angle:" + str(phase_angle.shape))
            spec = mag * np.exp(1.j * phase_angle)
            print("spec:" + str(spec.shape))
    return deemphasis(wav)


def _get_wav_from_mfccs(mfccs, preemphasis_coeff, n_fft, win_length, hop_length, n_wav):
    dctm = librosa.filters.dct(hp.Default.n_mfcc, hp.Default.n_mels)
    mel_basis = librosa.filters.mel(
        hp.Default.sr, hp.Default.n_fft, hp.Default.n_mels)
    #bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))
    mel_db = np.dot(dctm.T, mfccs.T)
    mel = db_to_amplitude(mel_db)
    recon_magsq = np.dot(mel_basis.T, mel)
    # bin_scaling[:, np.newaxis] *
    mag = np.sqrt(recon_magsq)
    excitation = np.random.randn(n_wav)
    E = librosa.stft(excitation, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    recon = librosa.core.istft(np.sqrt(recon_stft), hop_length=hop_length, win_length=win_length)
    recon = spectrogram2wav(mag, n_fft, win_length,
                            hop_length, hp.Default.n_iter)
    recon = deemphasis(recon, coeff=preemphasis_coeff)
    return recon


wav_file = "datasets/arctic/bdl/arctic_a0001.wav"
wav, sr = librosa.load(wav_file, sr=hp.Default.sr)
mfccs, mag, _ = _get_mfcc_log_spec_and_log_mel_spec(wav,
                                                    hp.Default.preemphasis,
                                                    hp.Default.n_fft,
                                                    hp.Default.win_length,
                                                    hp.Default.hop_length)
print(mag)
print (mag.shape)
audio = _get_wav_from_mfccs(mfccs,
                            hp.Default.preemphasis,
                            hp.Default.n_fft,
                            hp.Default.win_length,
                            hp.Default.hop_length,
                            len(wav))
audio2 = spectrogram2wav((np.e**mag).T,
                         hp.Default.n_fft,
                         hp.Default.win_length,
                         hp.Default.hop_length,
                         hp.Default.n_iter)
librosa.output.write_wav("recon2.wav", audio2, hp.Default.sr, norm=True)
