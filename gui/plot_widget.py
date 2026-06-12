"""pyqtgraph-based 2-D fish track plot with per-file replay support.

Axis convention:
  horizontal → tank Y  (0 … 120 cm)
  vertical   → tank X  (0 at top … 80 cm at bottom)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QVBoxLayout

_PALETTE = [
    "#e06c75", "#98c379", "#e5c07b", "#61afef",
    "#c678dd", "#56b6c2", "#d19a66", "#abb2bf",
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff",
]


def _color(fish_id: int) -> tuple:
    c = QColor(_PALETTE[fish_id % len(_PALETTE)])
    return (c.red(), c.green(), c.blue(), 220)


# One stored observation
_Obs = tuple   # (fish_id, x, y, freq, std_x, std_y)


class TrackPlotWidget(QWidget):
    """2-D fish track plot.

    Call ``record(file_idx, result)`` as each file finishes.
    Call ``show_up_to(file_idx)`` to replay up to that minute.
    """

    def __init__(self, xlim=(0, 80), ylim=(0, 120), parent=None):
        super().__init__(parent)
        self._xlim = xlim   # tank X range
        self._ylim = ylim   # tank Y range

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#0d0d2a")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._glw)

        self._plot = self._glw.addPlot(title="Fish Tracks")
        self._plot.setLabel("bottom", "Y (cm)")
        self._plot.setLabel("left",   "X (cm)")
        self._plot.setXRange(*ylim, padding=0)
        self._plot.setYRange(*xlim, padding=0)
        self._plot.setLimits(xMin=ylim[0], xMax=ylim[1],
                              yMin=xlim[0], yMax=xlim[1])
        self._plot.invertY(True)                    # 0 at top-left
        self._plot.disableAutoRange()
        self._plot.showGrid(x=True, y=True, alpha=0.2)

        self._sensor_scatter = pg.ScatterPlotItem(
            symbol="t", size=12,
            brush=pg.mkBrush(180, 180, 180, 160),
            pen=pg.mkPen(None),
        )
        self._plot.addItem(self._sensor_scatter)

        # All recorded observations: file_idx → list[_Obs]
        self._history: dict[int, list[_Obs]] = defaultdict(list)
        self._max_file_idx: int = 0

        # Live plot items per fish id
        self._tracks: dict = {}
        self._legend = self._plot.addLegend(offset=(10, 10))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_sensors(self, xy_meas: np.ndarray) -> None:
        self._sensor_scatter.setData(
            x=xy_meas[:, 1].tolist(),   # tank Y → horizontal
            y=xy_meas[:, 0].tolist(),   # tank X → vertical
        )

    def record(self, file_idx: int, result) -> None:
        """Store observations from *result* at *file_idx* and plot them."""
        for fish in result.fish:
            obs: _Obs = (
                fish["id"], fish["x"], fish["y"], fish["freq"],
                fish.get("std_x", 0.0), fish.get("std_y", 0.0),
            )
            self._history[file_idx].append(obs)
        self._max_file_idx = max(self._max_file_idx, file_idx)
        # Paint new points immediately (live mode)
        for obs in self._history[file_idx]:
            self._paint_obs(obs)

    def show_up_to(self, file_idx: int) -> None:
        """Replay all observations for files 0 … file_idx."""
        self._clear_plot_items()
        for idx in sorted(self._history.keys()):
            if idx > file_idx:
                break
            for obs in self._history[idx]:
                self._paint_obs(obs)

    def clear_all(self) -> None:
        self._clear_plot_items()
        self._history.clear()
        self._max_file_idx = 0

    @property
    def max_file_idx(self) -> int:
        return self._max_file_idx

    # kept for backward compat (used by old signal path)
    def update_from_result(self, result, file_idx: int = 0) -> None:
        self.record(file_idx, result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _paint_obs(self, obs: _Obs) -> None:
        fish_id, x, y, freq, std_x, std_y = obs
        col = _color(fish_id)

        if fish_id not in self._tracks:
            pen    = pg.mkPen(color=col, width=2)
            brush  = pg.mkBrush(color=col)
            line   = pg.PlotDataItem(pen=pen, name=f"Fish {fish_id}")
            scatter = pg.ScatterPlotItem(symbol="o", size=8,
                                         brush=brush, pen=pg.mkPen(None))
            self._plot.addItem(line)
            self._plot.addItem(scatter)
            self._tracks[fish_id] = {
                "line": line, "scatter": scatter,
                "pys": [], "pxs": [],   # screen coords
                "err_items": [],
            }

        entry = self._tracks[fish_id]
        entry["pys"].append(y)   # tank Y → screen horizontal
        entry["pxs"].append(x)   # tank X → screen vertical

        pys, pxs = entry["pys"], entry["pxs"]
        entry["line"].setData(x=pys, y=pxs)
        entry["scatter"].setData(x=[pys[-1]], y=[pxs[-1]])

        if std_x > 0 or std_y > 0:
            r, g, b, _ = col
            ep = pg.mkPen(color=(r, g, b, 100), width=1, style=Qt.DashLine)
            eh = pg.PlotDataItem(x=[y - std_y, y + std_y], y=[x,       x      ], pen=ep)
            ev = pg.PlotDataItem(x=[y,         y        ], y=[x - std_x, x + std_x], pen=ep)
            self._plot.addItem(eh)
            self._plot.addItem(ev)
            entry["err_items"].extend([eh, ev])

    def _clear_plot_items(self) -> None:
        for entry in self._tracks.values():
            self._plot.removeItem(entry["line"])
            self._plot.removeItem(entry["scatter"])
            for item in entry["err_items"]:
                self._plot.removeItem(item)
        self._tracks.clear()


# ---------------------------------------------------------------------------
# Frequency vs file-index plot
# ---------------------------------------------------------------------------

class FreqPlotWidget(QWidget):
    """Fish EOD frequency vs file index, colour-coded by fish ID."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: dict[int, list] = defaultdict(list)  # file_idx → [(fish_id, freq)]
        self._max_file_idx: int = 0
        self._tracks: dict = {}   # fish_id → {line, scatter, xs, ys}

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#0d0d2a")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._glw)

        self._plot = self._glw.addPlot(title="Frequency vs File")
        self._plot.setLabel("bottom", "File #")
        self._plot.setLabel("left",   "Frequency (Hz)")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setMenuEnabled(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, file_idx: int, result) -> None:
        for fish in result.fish:
            self._history[file_idx].append((fish["id"], fish["freq"]))
        self._max_file_idx = max(self._max_file_idx, file_idx)
        for fish_id, freq in self._history[file_idx]:
            self._paint(file_idx, fish_id, freq)

    def show_up_to(self, file_idx: int) -> None:
        self._clear_items()
        for idx in sorted(self._history.keys()):
            if idx > file_idx:
                break
            for fish_id, freq in self._history[idx]:
                self._paint(idx, fish_id, freq)

    def clear_all(self) -> None:
        self._clear_items()
        self._history.clear()
        self._max_file_idx = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _paint(self, file_idx: int, fish_id: int, freq: float) -> None:
        col = _color(fish_id)
        if fish_id not in self._tracks:
            pen    = pg.mkPen(color=col, width=2)
            brush  = pg.mkBrush(color=col)
            line   = pg.PlotDataItem(pen=pen, name=f"Fish {fish_id}")
            scatter = pg.ScatterPlotItem(symbol="o", size=7,
                                          brush=brush, pen=pg.mkPen(None))
            self._plot.addItem(line)
            self._plot.addItem(scatter)
            self._tracks[fish_id] = {"line": line, "scatter": scatter,
                                      "xs": [], "ys": []}
        entry = self._tracks[fish_id]
        entry["xs"].append(file_idx + 1)   # 1-based for display
        entry["ys"].append(freq)
        entry["line"].setData(x=entry["xs"], y=entry["ys"])
        entry["scatter"].setData(x=entry["xs"], y=entry["ys"])

    def _clear_items(self) -> None:
        for entry in self._tracks.values():
            self._plot.removeItem(entry["line"])
            self._plot.removeItem(entry["scatter"])
        self._tracks.clear()
