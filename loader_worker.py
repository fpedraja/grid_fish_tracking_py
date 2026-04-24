"""Lightweight worker that loads and filters a single WAV file on demand
so the signal viewer can be updated without blocking the GUI."""

from __future__ import annotations

import traceback

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from pipeline import FishTrackingConfig, _pick_snippet
from core.signal_proc import (bandpass_filter, compute_envelope,
                               detect_peaks_all_channels, merge_event_times)


class SignalLoaderWorker(QThread):
    """Read + filter one WAV file and emit the result."""

    signal_ready = pyqtSignal(float, object, object, object, str)  # fs, snippet, et_sec, labels|None, fname
    error        = pyqtSignal(str)

    def __init__(self, fpath: str, fname: str,
                 config: FishTrackingConfig, parent=None):
        super().__init__(parent)
        self._fpath  = fpath
        self._fname  = fname
        self._config = config

    def run(self) -> None:
        try:
            import soundfile as sf
            import scipy.io.wavfile as wv

            try:
                data, fs = sf.read(self._fpath)
            except Exception:
                fs, data = wv.read(self._fpath)
                data = data.astype(np.float32)
                if data.max() > 1.0:
                    data = data / 32768.0

            if data.ndim == 1:
                data = data[:, np.newaxis]

            fs = float(fs)
            filtered  = bandpass_filter(data, fs,
                                        self._config.bp_low,
                                        self._config.bp_high,
                                        self._config.bp_order)
            env       = compute_envelope(filtered)
            raw       = detect_peaks_all_channels(env, fs,
                                                  self._config.min_pk_height,
                                                  self._config.mpd_ms)
            events    = merge_event_times(raw, fs, self._config.merge_ms)

            i0, i1 = _pick_snippet(events, fs, filtered.shape[0])
            win_mask = (events >= i0) & (events < i1)
            snippet = filtered[i0:i1, :]
            et_sec  = (events[win_mask] - i0) / fs
            self.signal_ready.emit(float(fs), snippet, et_sec, None, self._fname)
        except Exception:
            self.error.emit(traceback.format_exc())
