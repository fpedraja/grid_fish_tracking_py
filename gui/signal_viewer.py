"""1-second EOD signal viewer with cluster-colour-coded detection markers."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

_CH_COLORS = [
    '#e06c75', '#98c379', '#e5c07b', '#61afef',
    '#c678dd', '#56b6c2', '#d19a66', '#abb2bf',
]

_CLUSTER_PALETTE = [
    '#e06c75', '#98c379', '#e5c07b', '#61afef',
    '#c678dd', '#56b6c2', '#d19a66', '#ff6b6b',
    '#ffd93d', '#6bcb77', '#4d96ff',
]


def _cluster_brush(label: int) -> pg.mkBrush:
    from PyQt5.QtGui import QColor
    if label < 0:
        return pg.mkBrush(160, 160, 160, 100)   # noise → grey
    c = QColor(_CLUSTER_PALETTE[(label - 1) % len(_CLUSTER_PALETTE)])
    return pg.mkBrush(c.red(), c.green(), c.blue(), 220)


class SignalViewer(QWidget):
    """8-channel bandpass signal – best 1-second window per file.

    Call ``set_data(fs, snippet, event_times_sec, event_labels, fname)``
    whenever a new file is available.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fs: float = 20000.0
        self._data: np.ndarray | None = None
        self._events_sec: np.ndarray | None = None
        self._event_labels: np.ndarray | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, fs: float, snippet: np.ndarray,
                 event_times_sec: np.ndarray,
                 event_labels=None, fname: str = "") -> None:
        """Show one second of data.

        Args:
            fs:              sample rate of snippet (Hz)
            snippet:         (N × 8) filtered signal for ~1 second
            event_times_sec: event times in seconds (relative to snippet start)
            event_labels:    cluster labels per event (None → all grey)
            fname:           filename shown in header
        """
        self._fs = fs
        self._data = snippet
        self._events_sec = np.asarray(event_times_sec) if event_times_sec is not None else np.array([])
        self._event_labels = (np.asarray(event_labels, dtype=int)
                              if event_labels is not None else None)
        self._file_label.setText(fname)
        self._render()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(1)

        self._file_label = QLabel("No data")
        self._file_label.setStyleSheet("color:#abb2bf; font-style:italic; font-size:8pt;")
        root.addWidget(self._file_label)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#0d0d2a")
        self._glw.ci.setSpacing(1)
        self._glw.ci.setContentsMargins(4, 0, 4, 0)
        root.addWidget(self._glw, stretch=1)

        self._plots:       list[pg.PlotItem] = []
        self._curves:      list[pg.PlotDataItem] = []
        self._ev_items:    list[list] = []   # per-channel list of ScatterPlotItems

        link_to = None
        for i in range(8):
            col = _CH_COLORS[i]
            p = self._glw.addPlot(row=i, col=0)
            p.setLabel('left', f'CH{i+1}', color=col, **{'font-size': '8pt'})
            p.showGrid(x=True, y=False, alpha=0.15)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.getAxis('left').setWidth(36)

            if i < 7:
                p.getAxis('bottom').setStyle(showValues=False)
                p.getAxis('bottom').setHeight(0)
            else:
                p.setLabel('bottom', 'Time (s)')

            if link_to is not None:
                p.setXLink(link_to)
            else:
                link_to = p

            curve = p.plot(pen=pg.mkPen(color=col, width=1))
            curve.setClipToView(True)

            self._plots.append(p)
            self._curves.append(curve)
            self._ev_items.append([])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self._data is None:
            return

        n_ch = min(8, self._data.shape[1])
        n    = self._data.shape[0]
        t_arr = np.arange(n) / self._fs

        # Waveforms
        for ch in range(n_ch):
            self._curves[ch].setData(x=t_arr, y=self._data[:, ch])
            self._plots[ch].setXRange(t_arr[0], t_arr[-1], padding=0.02)

        # Remove old event scatter items
        for ch in range(n_ch):
            for item in self._ev_items[ch]:
                self._plots[ch].removeItem(item)
            self._ev_items[ch].clear()

        if self._events_sec is None or len(self._events_sec) == 0:
            return

        et  = self._events_sec
        lbl = (self._event_labels if self._event_labels is not None
               else np.ones(len(et), dtype=int))

        # Clip event times to snippet range
        mask = (et >= 0) & (et <= t_arr[-1])
        et   = et[mask]
        lbl  = lbl[mask]
        if len(et) == 0:
            return

        # Add one ScatterPlotItem per unique cluster per channel
        unique_labels = np.unique(lbl)
        for ch in range(n_ch):
            ch_data = self._data[:, ch]
            for cluster_lbl in unique_labels:
                cmask  = lbl == cluster_lbl
                ev_t   = et[cmask]
                ev_i   = np.clip((ev_t * self._fs).astype(int), 0, n - 1)
                ev_y   = ch_data[ev_i]
                sc = pg.ScatterPlotItem(
                    x=ev_t, y=ev_y,
                    symbol='t', size=10,
                    brush=_cluster_brush(int(cluster_lbl)),
                    pen=pg.mkPen(None),
                )
                self._plots[ch].addItem(sc)
                self._ev_items[ch].append(sc)
