import numpy as np


def minmax_scale(x, xmin, xmax):
    """Min-max normalization."""
    return (x - xmin) / (xmax - xmin + 1e-8)


def minmax_inverse(x_norm, xmin, xmax):
    """Inverse min-max normalization."""
    return x_norm * (xmax - xmin) + xmin


def standardize(x):
    """Standard score normalization."""
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def smooth_signal(arr, factor=0.9):
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = factor * smoothed[i - 1] + (1 - factor) * arr[i]
    return smoothed


def clip_features(features, low, high):
    """Clip features to specified bounds."""
    return np.clip(features, low, high)
