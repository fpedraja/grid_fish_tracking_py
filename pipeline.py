"""Main processing pipeline: ties together signal processing, localization,
clustering, and tracking for a folder of 1-minute WAV recordings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.io.wavfile as _wavfile

from core.signal_proc import (bandpass_filter, comb_notch_filter,
                               compute_envelope,
                               detect_peaks_all_channels, merge_event_times,
                               amplitude_snapshots)
from core.localization import (build_spatial_grid, precompute_spatial_weights,
                                localize_events)
from core.clustering import (cluster_events_dbscan, compute_cluster_stats,
                              compute_amp_fingerprints, filter_valid_clusters)
from core.kalman_tracker import KalmanFishTracker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FishTrackingConfig:
    """All tunable parameters for the fish-tracking pipeline."""

    # --- Sensor geometry ---
    xy_meas: np.ndarray = field(default_factory=lambda: np.array([
        [ 0,   0],   # ch1
        [ 0,  80],   # ch2
        [40,   0],   # ch3
        [40,  40],   # ch4
        [40,  80],   # ch5
        [40, 120],   # ch6
        [80,  40],   # ch7
        [80, 120],   # ch8
    ], dtype=float))

    # --- Spatial grid ---
    xlim:          tuple = (0, 80)
    ylim:          tuple = (0, 120)
    grid_step:     float = 1.0       # cm resolution
    sigma_spatial: float = 30.0      # Gaussian kernel width (cm)

    # --- Bandpass filter ---
    bp_low:  float = 300.0   # Hz
    bp_high: float = 2000.0  # Hz
    bp_order: int  = 3

    # --- Peak detection ---
    min_pk_height: float = 0.015  # envelope amplitude threshold
    mpd_ms:        float = 5.0    # minimum peak distance (ms)
    merge_ms:      float = 2.0    # cross-channel merge window (ms)

    # --- DBSCAN ---
    eps_phys: float = 20.0   # neighbourhood radius (cm)
    min_pts:  int   = 60     # minimum events per cluster

    # --- Cluster validity ---
    min_events_for_fish:    int   = 600   # per-file gate (before clustering)
    min_events_per_cluster: int   = 200   # per-cluster gate
    min_dur_per_cluster:    float = 5.0   # seconds
    min_freq_hz:            float = 5.0   # Hz
    max_freq_hz:            float = 120.0 # Hz

    # --- Tracker ---
    dt: float = 60.0   # seconds between files

    # --- Tracker noise params ---
    meas_sig_pos:  float = 6.0
    meas_sig_f:    float = 0.15
    wander_1min:   float = 40.0
    freq_drift_1m: float = 0.3

    # --- Tracker association gates ---
    hard_gate_pos: float = 60.0   # cm
    hard_gate_f:   float = 2.0    # Hz  ← most common reason same fish gets split
    sig_gate:      float = 0.35   # cosine distance (0=identical, 1=orthogonal)
    max_miss:      int   = 10     # files a track can coast without a detection

    # --- Notch filter ---
    notch_enabled: bool  = False
    notch_hz:      float = 50.0   # powerline fundamental (Hz) — Uruguay: 50 Hz
    notch_Q:       float = 30.0   # quality factor (higher = narrower notch)


# ---------------------------------------------------------------------------
# Per-file result
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    filename:   str
    fs:         float
    n_events:   int
    n_clusters: int
    fish: list     # list[dict]: keys id, x, y, freq


# ---------------------------------------------------------------------------
# Progress callback signature
# ---------------------------------------------------------------------------
# progress_cb(file_idx, total_files, filename, message_or_None, result_or_None)

ProgressCB = Optional[Callable]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _pick_snippet(event_times: np.ndarray, fs: float, n_samples: int) -> tuple:
    """Return (i0, i1) of the 1-second window containing the most events."""
    if len(event_times) == 0:
        return 0, min(n_samples, int(fs))
    n_wins = max(1, int(n_samples / fs))
    t_sec = event_times / fs
    counts, _ = np.histogram(t_sec, bins=np.arange(0, n_wins + 1))
    best = int(np.argmax(counts))
    i0 = int(best * fs)
    i1 = min(n_samples, int((best + 1) * fs))
    return i0, i1


def process_folder(folder_path: str,
                   config: FishTrackingConfig,
                   progress_cb: ProgressCB = None,
                   signal_cb:   ProgressCB = None,
                   cluster_cb:  ProgressCB = None,
                   stop_flag:   Optional[Callable] = None
                   ) -> tuple[list[FileResult], pd.DataFrame]:
    """Process all WAV files in *folder_path*.

    Args:
        folder_path:  path to directory containing 1-minute WAV files
        config:       FishTrackingConfig instance
        progress_cb:  optional callback(file_idx, total, fname, msg, result)
        stop_flag:    optional callable returning True to abort early

    Returns:
        (results, df) where *results* is a list of FileResult objects and
        *df* is a pandas DataFrame with columns
        [file_idx, filename, track_id, x_cm, y_cm, freq_hz].
    """
    wav_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".wav")
    )
    if not wav_files:
        raise ValueError(f"No WAV files found in: {folder_path}")

    n_ch = len(config.xy_meas)

    # Pre-compute spatial structures (done once, shared across all files)
    xy_grid  = build_spatial_grid(config.xlim, config.ylim, config.grid_step)
    W_proto  = precompute_spatial_weights(config.xy_meas, xy_grid, config.sigma_spatial)

    tracker = KalmanFishTracker(
        meas_sig_pos  = config.meas_sig_pos,
        meas_sig_f    = config.meas_sig_f,
        wander_1min   = config.wander_1min,
        freq_drift_1m = config.freq_drift_1m,
        xlim          = config.xlim,
        ylim          = config.ylim,
        hard_gate_pos = config.hard_gate_pos,
        hard_gate_f   = config.hard_gate_f,
        sig_gate      = config.sig_gate,
        max_miss      = config.max_miss,
    )

    results:    list[FileResult] = []
    track_rows: list[dict]       = []

    for file_idx, fname in enumerate(wav_files):
        if stop_flag and stop_flag():
            break

        fpath = os.path.join(folder_path, fname)
        if os.path.getsize(fpath) == 0:
            _cb(progress_cb, file_idx, len(wav_files), fname, "Empty file – skipped", None)
            continue
        _cb(progress_cb, file_idx, len(wav_files), fname, None, None)

        # ---- Load audio (soundfile first; fall back to scipy for non-standard WAVs) ----
        try:
            data, fs = sf.read(fpath)
        except Exception:
            try:
                fs, data = _wavfile.read(fpath)
                data = data.astype(np.float32)
                if data.max() > 1.0:
                    data /= np.iinfo(np.int16).max  # normalise int16 → [-1, 1]
            except Exception as exc:
                _cb(progress_cb, file_idx, len(wav_files), fname,
                    f"Read error: {exc}", None)
                continue

        if data.ndim == 1:
            data = data[:, np.newaxis]

        if data.shape[1] != n_ch:
            _cb(progress_cb, file_idx, len(wav_files), fname,
                f"Channel mismatch: got {data.shape[1]}, expected {n_ch}", None)
            continue

        # ---- Signal processing ----
        if config.notch_enabled:
            data = comb_notch_filter(data, fs, config.notch_hz, config.notch_Q)
        filtered = bandpass_filter(data, fs, config.bp_low,
                                   config.bp_high, config.bp_order)
        env = compute_envelope(filtered)

        # ---- Peak detection ----
        raw_times  = detect_peaks_all_channels(env, fs,
                                               config.min_pk_height,
                                               config.mpd_ms)
        event_times = merge_event_times(raw_times, fs, config.merge_ms)

        if len(event_times) < config.min_events_for_fish:
            _cb(progress_cb, file_idx, len(wav_files), fname,
                f"Too few events: {len(event_times)}", None)
            if signal_cb is not None:
                i0, i1 = _pick_snippet(event_times, fs, filtered.shape[0])
                win_mask = (event_times >= i0) & (event_times < i1)
                _cb(signal_cb, float(fs), filtered[i0:i1, :],
                    (event_times[win_mask] - i0) / fs, None, fname)
            continue

        # ---- Localization ----
        snaps    = amplitude_snapshots(env, event_times)

        # Signal viewer: best 1-second snippet (pre-cluster; labels=None)
        if signal_cb is not None:
            i0, i1 = _pick_snippet(event_times, fs, filtered.shape[0])
            win_mask = (event_times >= i0) & (event_times < i1)
            _cb(signal_cb, float(fs), filtered[i0:i1, :],
                (event_times[win_mask] - i0) / fs, None, fname)

        X, Y     = localize_events(snaps, W_proto, xy_grid)

        # ---- DBSCAN clustering ----
        labels = cluster_events_dbscan(X, Y, config.eps_phys, config.min_pts)
        if not np.any(labels >= 0):
            _cb(progress_cb, file_idx, len(wav_files), fname,
                "No clusters found", None)
            continue

        # ---- Per-cluster stats + fingerprints ----
        stats        = compute_cluster_stats(event_times, X, Y, labels, fs)
        fingerprints = compute_amp_fingerprints(snaps, labels)
        valid_stats, valid_fps = filter_valid_clusters(
            stats, fingerprints,
            config.min_events_per_cluster,
            config.min_dur_per_cluster,
            config.min_freq_hz,
            config.max_freq_hz,
        )

        if not valid_stats:
            _cb(progress_cb, file_idx, len(wav_files), fname,
                "No valid clusters after filtering", None)
            continue

        # Update signal viewer with cluster-colour-coded snippet
        if signal_cb is not None:
            i0, i1 = _pick_snippet(event_times, fs, filtered.shape[0])
            win_mask = (event_times >= i0) & (event_times < i1)
            _cb(signal_cb, float(fs), filtered[i0:i1, :],
                (event_times[win_mask] - i0) / fs, labels[win_mask], fname)

        # UMAP/PCA cluster identity scatter
        if cluster_cb is not None:
            valid_label_set = {s["label"] for s in valid_stats}
            fp_mask = np.isin(labels, sorted(valid_label_set))
            _cb(cluster_cb, snaps[fp_mask], labels[fp_mask], fname)

        # Sort by frequency descending (matches MATLAB behaviour)
        order       = np.argsort([-s["freq"] for s in valid_stats])
        valid_stats = [valid_stats[i] for i in order]
        valid_fps   = [valid_fps[i]   for i in order]

        centroids = np.array([[s["centroid_x"], s["centroid_y"]]
                               for s in valid_stats])
        freqs     = np.array([s["freq"] for s in valid_stats])

        # ---- Kalman tracker update ----
        tracks  = tracker.update(centroids, freqs, valid_fps, config.dt)
        matched = [t for t in tracks if t.miss == 0]

        # Emit per-association log lines via progress callback
        for line in getattr(tracker, "last_log", []):
            _cb(progress_cb, file_idx, len(wav_files), fname, line, None)

        fish_list = []
        for t in matched:
            # Find nearest cluster to get raw position std
            dists = [np.hypot(t.x[0] - s["centroid_x"], t.x[1] - s["centroid_y"])
                     for s in valid_stats]
            nearest = valid_stats[int(np.argmin(dists))]
            fish_list.append({
                "id":    t.id,
                "x":     float(t.x[0]),
                "y":     float(t.x[1]),
                "freq":  float(t.x[4]),
                "std_x": nearest["std_x"],
                "std_y": nearest["std_y"],
            })
            track_rows.append({
                "file_idx":  file_idx + 1,
                "filename":  fname,
                "track_id":  t.id,
                "x_cm":      float(t.x[0]),
                "y_cm":      float(t.x[1]),
                "freq_hz":   float(t.x[4]),
                "std_x_cm":  nearest["std_x"],
                "std_y_cm":  nearest["std_y"],
            })

        result = FileResult(
            filename=fname, fs=float(fs),
            n_events=len(event_times),
            n_clusters=len(valid_stats),
            fish=fish_list,
        )
        results.append(result)
        _cb(progress_cb, file_idx, len(wav_files), fname, None, result)

    cols = ["file_idx", "filename", "track_id", "x_cm", "y_cm", "freq_hz", "std_x_cm", "std_y_cm"]
    df   = pd.DataFrame(track_rows) if track_rows else pd.DataFrame(columns=cols)

    return results, df


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cb(fn, *args):
    if fn is not None:
        try:
            fn(*args)
        except Exception:
            pass
