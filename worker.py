"""Background QThread worker that runs the processing pipeline without
blocking the GUI event loop."""

from __future__ import annotations

import traceback

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from pipeline import FishTrackingConfig, FileResult, process_folder
import pandas as pd


class ProcessingWorker(QThread):
    """Runs process_folder() in a background thread.

    Signals
    -------
    progress(file_idx, total, filename, message)
    file_done(result)
    signal_ready(fs, filtered_data, event_times, filename)
        Emitted after each file's signal processing step with downsampled data.
    finished_ok(df)
    error(msg)
    """

    progress      = pyqtSignal(int, int, str, object)               # idx, total, fname, msg|None
    file_done     = pyqtSignal(object)                              # FileResult
    signal_ready  = pyqtSignal(float, object, object, object, str) # fs, snippet, et_sec, labels|None, fname
    cluster_ready = pyqtSignal(object, object, str)                 # snaps, labels, fname
    finished_ok   = pyqtSignal(object)                              # pd.DataFrame
    error         = pyqtSignal(str)

    def __init__(self, folder: str, config: FishTrackingConfig, parent=None):
        super().__init__(parent)
        self._folder  = folder
        self._config  = config
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def run(self) -> None:
        try:
            _, df = process_folder(
                self._folder,
                self._config,
                progress_cb=self._on_progress,
                signal_cb=self._on_signal,
                cluster_cb=self._on_cluster,
                stop_flag=lambda: self._stopped,
            )
            self.finished_ok.emit(df)
        except Exception:
            self.error.emit(traceback.format_exc())

    # ------------------------------------------------------------------
    def _on_progress(self, file_idx: int, total: int,
                     fname: str, msg, result: FileResult | None) -> None:
        self.progress.emit(file_idx, total, fname, msg)
        if result is not None:
            self.file_done.emit(result)

    def _on_signal(self, fs: float, snippet: np.ndarray,
                   event_times_sec: np.ndarray, labels, fname: str) -> None:
        self.signal_ready.emit(fs, snippet, event_times_sec, labels, fname)

    def _on_cluster(self, snaps: np.ndarray, labels: np.ndarray, fname: str) -> None:
        self.cluster_ready.emit(snaps, labels, fname)
