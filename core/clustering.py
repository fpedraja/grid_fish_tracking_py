"""DBSCAN clustering, per-cluster statistics, and amplitude fingerprints."""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_events_dbscan(X: np.ndarray, Y: np.ndarray,
                           eps_phys: float = 20.0,
                           min_pts: int = 60) -> np.ndarray:
    """Cluster localized EOD events with DBSCAN.

    Args:
        X, Y:      (E,) event positions in cm
        eps_phys:  neighbourhood radius in cm
        min_pts:   minimum cluster size

    Returns:
        labels: (E,) integer labels (-1 = noise).
    """
    coords = np.column_stack([Y, X])          # match MATLAB [Y X] convention
    return DBSCAN(eps=eps_phys, min_samples=min_pts).fit_predict(coords)


def compute_cluster_stats(event_times: np.ndarray,
                           X: np.ndarray, Y: np.ndarray,
                           labels: np.ndarray,
                           fs: float) -> list:
    """Compute per-cluster centroid, frequency, duration, and event count.

    Frequency is estimated as 1 / median(ISI) for robustness against
    outlier inter-spike intervals.

    Args:
        event_times: (E,) sample indices of merged events
        X, Y:        (E,) localized positions in cm
        labels:      (E,) DBSCAN labels
        fs:          sample rate (Hz)

    Returns:
        List of dicts with keys:
        label, count, duration, centroid_x, centroid_y, freq.
    """
    t_sec = event_times / fs
    unique_labels = np.unique(labels[labels >= 0])
    stats = []
    for k in unique_labels:
        mask = labels == k
        ev_t = np.sort(t_sec[mask])
        freq = np.nan
        if len(ev_t) >= 2:
            isis = np.diff(ev_t)
            med_isi = np.median(isis)
            if med_isi > 0:
                freq = 1.0 / med_isi
        stats.append({
            "label":      int(k),
            "count":      int(mask.sum()),
            "duration":   float(ev_t[-1] - ev_t[0]) if len(ev_t) >= 2 else 0.0,
            "centroid_x": float(np.median(X[mask])),
            "centroid_y": float(np.median(Y[mask])),
            "std_x":      float(np.std(X[mask])),
            "std_y":      float(np.std(Y[mask])),
            "freq":       float(freq),
        })
    return stats


def compute_amp_fingerprints(snaps: np.ndarray,
                              labels: np.ndarray,
                              min_events: int = 20) -> dict:
    """Build an L2-normalised amplitude fingerprint per cluster.

    Fourth-root compression (softer than cube-root) is used here so that
    the fingerprint captures relative channel activations faithfully.

    Args:
        snaps:      (E x C) amplitude snapshots
        labels:     (E,) DBSCAN labels
        min_events: minimum events required to form a fingerprint

    Returns:
        Dict mapping label (int) -> fingerprint (C,) array.
    """
    fingerprints = {}
    for k in np.unique(labels[labels >= 0]):
        mask = labels == k
        if mask.sum() < min_events:
            continue
        V = np.maximum(snaps[mask], 0.0) ** 0.25          # fourth-root compression
        row_sums = V.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = np.finfo(float).eps
        V /= row_sums                                      # per-event sum-normalise
        fp = np.median(V, axis=0)
        nrm = np.linalg.norm(fp)
        if nrm == 0:
            fp[0] = 1.0
            nrm = 1.0
        fingerprints[int(k)] = fp / nrm                   # L2-normalise
    return fingerprints


def filter_valid_clusters(stats: list, fingerprints: dict,
                           min_events: int = 200,
                           min_dur: float = 5.0,
                           min_freq: float = 5.0,
                           max_freq: float = 120.0) -> tuple:
    """Keep clusters that pass all quality thresholds.

    Args:
        stats:         output of compute_cluster_stats()
        fingerprints:  output of compute_amp_fingerprints()
        min_events:    minimum number of events in the cluster
        min_dur:       minimum duration spanned by the cluster (s)
        min_freq:      minimum discharge frequency (Hz)
        max_freq:      maximum discharge frequency (Hz)

    Returns:
        (valid_stats, valid_fps) – parallel lists of passing clusters.
    """
    valid_stats, valid_fps = [], []
    for s in stats:
        if (s["count"] >= min_events and
                s["duration"] >= min_dur and
                np.isfinite(s["freq"]) and
                min_freq <= s["freq"] <= max_freq):
            valid_stats.append(s)
            valid_fps.append(fingerprints.get(s["label"]))
    return valid_stats, valid_fps
