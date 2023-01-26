"""
Author: Frank Zalkow, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
import librosa

def stft_convention_fmp(x, Fs, N, H, pad_mode='constant', center=True, mag=False, gamma=0):
    """Compute the discrete short-time Fourier transform (STFT)
    Notebook: C2/C2_STFT-FreqGridInterpol.ipynb
    Args:
        x (np.ndarray): Signal to be transformed
        Fs (scalar): Sampling rate
        N (int): Window size
        H (int): Hopsize
        pad_mode (str): Padding strategy is used in librosa (Default value = 'constant')
        center (bool): Centric view as used in librosa (Default value = True)
        mag (bool): Computes magnitude STFT if mag==True (Default value = False)
        gamma (float): Constant for logarithmic compression (only applied when mag==True) (Default value = 0)
    Returns:
        X (np.ndarray): Discrete (magnitude) short-time Fourier transform
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N,
                     window='hann', pad_mode=pad_mode, center=center)
    if mag:
        X = np.abs(X)**2
        if gamma > 0:
            X = np.log(1 + gamma * X)
    F_coef = librosa.fft_frequencies(sr=Fs, n_fft=N)
    T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
    # T_coef = np.arange(X.shape[1]) * H/Fs
    # F_coef = np.arange(N//2+1) * Fs/N
    return X, T_coef, F_coef


def compute_f_coef_linear(N, Fs, rho=1):
    """Refines the frequency vector by factor of rho
    Notebook: C2/C2_STFT-FreqGridInterpol.ipynb
    Args:
        N (int): Window size
        Fs (scalar): Sampling rate
        rho (int): Factor for refinement (Default value = 1)
    Returns:
        F_coef_new (np.ndarray): Refined frequency vector
    """
    L = rho * N
    F_coef_new = np.arange(0, L//2+1) * Fs / L
    return F_coef_new


def compute_f_coef_log(R, F_min, F_max):
    """Adapts the frequency vector in a logarithmic fashion
    Notebook: C2/C2_STFT-FreqGridInterpol.ipynb
    Args:
        R (scalar): Resolution (cents)
        F_min (float): Minimum frequency
        F_max (float): Maximum frequency (not included)
    Returns:
        F_coef_log (np.ndarray): Refined frequency vector with values given in Hz)
        F_coef_cents (np.ndarray): Refined frequency vector with values given in cents.
            Note: F_min serves as reference (0 cents)
    """
    n_bins = np.ceil(1200 * np.log2(F_max / F_min) / R).astype(int)
    F_coef_log = 2 ** (np.arange(0, n_bins) * R / 1200) * F_min
    F_coef_cents = 1200 * np.log2(F_coef_log / F_min)
    return F_coef_log, F_coef_cents