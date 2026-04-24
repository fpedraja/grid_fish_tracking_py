"""Signal processing: filtering, envelope extraction, peak detection, event merging."""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, hilbert, find_peaks


def bandpass_filter(data: np.ndarray, fs: float,
                    low_hz: float = 300.0, high_hz: float = 2000.0,
                    order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        data:   (N x C) raw audio samples
        fs:     sample rate (Hz)
        low_hz: lower cutoff (Hz)
        high_hz: upper cutoff (Hz)
        order:  filter order

    Returns:
        Filtered data (N x C).
    """
    nyq = fs / 2.0
    Wn = [low_hz / nyq, high_hz / nyq]
    b, a = butter(order, Wn, btype="bandpass")
    return lfilter(b, a, data, axis=0)


def compute_envelope(filtered: np.ndarray) -> np.ndarray:
    """Hilbert envelope, squared, baseline-subtracted, non-negative.

    Squaring pushes signal energy well above noise and compresses the
    dynamic range for the downstream localization step.

    Args:
        filtered: (N x C) bandpass-filtered data

    Returns:
        Envelope (N x C), non-negative.
    """
    env = np.abs(hilbert(filtered, axis=0))            # (N x C)
    env -= np.median(env, axis=0, keepdims=True)       # per-channel baseline
    np.maximum(env, 0.0, out=env)
    env **= 2                                          # square to separate signal from noise
    return env


def detect_peaks_all_channels(env: np.ndarray, fs: float,
                               min_peak_height: float,
                               mpd_ms: float) -> np.ndarray:
    """Detect peaks on every channel and pool sample indices.

    Args:
        env:              (N x C) envelope
        fs:               sample rate (Hz)
        min_peak_height:  minimum envelope value to count as a peak
        mpd_ms:           minimum peak distance in milliseconds

    Returns:
        Sorted array of sample indices (int64).
    """
    mpd = max(1, int(round(mpd_ms / 1000.0 * fs)))
    all_locs = []
    for c in range(env.shape[1]):
        locs, _ = find_peaks(env[:, c], height=min_peak_height, distance=mpd)
        all_locs.append(locs)
    if not all_locs:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(all_locs).astype(np.int64))


def merge_event_times(all_times: np.ndarray, fs: float,
                      merge_ms: float) -> np.ndarray:
    """Remove near-simultaneous duplicate detections across channels.

    Args:
        all_times: sorted sample indices from all channels
        fs:        sample rate (Hz)
        merge_ms:  merge window in milliseconds

    Returns:
        De-duplicated sorted array of sample indices.
    """
    if len(all_times) == 0:
        return all_times
    merge_samp = max(1, int(round(merge_ms / 1000.0 * fs)))
    diffs = np.diff(all_times)
    keep = np.concatenate([[True], diffs >= merge_samp])
    return all_times[keep]


def amplitude_snapshots(env: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """Extract multi-channel envelope amplitude at each event.

    Args:
        env:         (N x C) envelope
        event_times: (E,) sample indices

    Returns:
        (E x C) amplitude matrix.
    """
    return env[event_times, :]
