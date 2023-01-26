"""
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
from scipy import signal
import scipy
from .plot_tools import *

def f_pitch(p, pitch_ref=69, freq_ref=440.0):
    """Computes the center frequency/ies of a MIDI pitch
    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb
    Args:
        p (float): MIDI pitch value(s)
        pitch_ref (float): Reference pitch (default: 69)
        freq_ref (float): Frequency of reference pitch (default: 440.0)
    Returns:
        freqs (float): Frequency value(s)
    """
    return 2 ** ((p - pitch_ref) / 12) * freq_ref

def log_compression(v, gamma=1.0):
    """Logarithmically compresses a value or array
    Notebook: C3/C3S1_LogCompression.ipynb
    Args:
        v (float or np.ndarray): Value or array
        gamma (float): Compression factor (Default value = 1.0)
    Returns:
        v_compressed (float or np.ndarray): Compressed value or array
    """
    return np.log(1 + gamma * v)


def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence
    Notebook: C3/C3S1_FeatureNormalization.ipynb
    Args:
        X (np.ndarray): Feature sequence
        norm (str): The norm to be applied. '1', '2', 'max' or 'z' (Default value = '2')
        threshold (float): An threshold below which the vector ``v`` used instead of normalization
            (Default value = 0.0001)
        v (float): Used instead of normalization below ``threshold``. If None, uses unit vector for given norm
            (Default value = None)
    Returns:
        X_norm (np.ndarray): Normalized feature sequence
    """
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm


def smooth_downsample_feature_sequence(X, Fs, filt_len=41, down_sampling=10, w_type='boxcar'):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by convolution with a filter kernel
    Notebook: C3/C3S1_FeatureSmoothing.ipynb
    Args:
        X (np.ndarray): Feature sequence
        Fs (scalar): Frame rate of ``X``
        filt_len (int): Length of smoothing filter (Default value = 41)
        down_sampling (int): Downsampling factor (Default value = 10)
        w_type (str): Window type of smoothing filter (Default value = 'boxcar')
    Returns:
        X_smooth (np.ndarray): Smoothed and downsampled feature sequence
        Fs_feature (scalar): Frame rate of ``X_smooth``
    """
    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature


def median_downsample_feature_sequence(X, Fs, filt_len=41, down_sampling=10):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by median filtering
    Notebook: C3/C3S1_FeatureSmoothing.ipynb
    Args:
        X (np.ndarray): Feature sequence
        Fs (scalar): Frame rate of ``X``
        filt_len (int): Length of smoothing filter (Default value = 41)
        down_sampling (int): Downsampling factor (Default value = 10)
    Returns:
        X_smooth (np.ndarray): Smoothed and downsampled feature sequence
        Fs_feature (scalar): Frame rate of ``X_smooth``
    """
    assert filt_len % 2 == 1  # L needs to be odd
    filt_len = [1, filt_len]
    X_smooth = signal.medfilt2d(X, filt_len)
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature

#--

#---

def compute_cost_matrix(X, Y, metric='euclidean'):
    """Compute the cost matrix of two feature sequences
    Notebook: C3/C3S2_DTWbasic.ipynb
    Args:
        X (np.ndarray): Sequence 1
        Y (np.ndarray): Sequence 2
        metric (str): Cost metric, a valid strings for scipy.spatial.distance.cdist (Default value = 'euclidean')
    Returns:
        C (np.ndarray): Cost matrix
    """
    X, Y = np.atleast_2d(X, Y)
    C = scipy.spatial.distance.cdist(X.T, Y.T, metric=metric)
    return C


def plot_matrix_with_points(C, P=np.empty((0, 2)), color='r', marker='o', linestyle='', **kwargs):
    """Compute the cost matrix of two feature sequences
    Args:
        C: Matrix to be plotted
        P: List of index pairs, to be visualized on the matrix (Default value = np.empty((0, 2)))
        color: The color of the line plot (Default value = 'r').
            See https://matplotlib.org/users/colors.html
        marker: The marker of the line plot (Default value = 'o').
            See https://matplotlib.org/3.1.0/api/markers_api.html
        linestyle: The line-style of the line plot (Default value = '').
            See https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
        **kwargs: keyword arguments for :func:`libfmp.b.b_plot.plot_matrix`
    Returns:
        fig: Handle for figure
        im: Handle for imshow
        line: handle for line plot
    """

    fig, ax, im = plot_matrix(C, **kwargs)
    line = ax[0].plot(P[:, 1], P[:, 0], marker=marker, color=color, linestyle=linestyle)

    if fig is not None:
        plt.tight_layout()

    return fig, im, line