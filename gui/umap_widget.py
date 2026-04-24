"""Cluster-identity scatter plot: amplitude fingerprints in 2-D via UMAP or PCA."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

_PALETTE = [
    "#e06c75", "#98c379", "#e5c07b", "#61afef",
    "#c678dd", "#56b6c2", "#d19a66", "#ff6b6b",
    "#ffd93d", "#6bcb77", "#4d96ff", "#abb2bf",
]


def _rgba(label: int) -> tuple:
    from PyQt5.QtGui import QColor
    c = QColor(_PALETTE[label % len(_PALETTE)])
    return (c.red(), c.green(), c.blue(), 180)


def _normalise_snaps(snaps: np.ndarray) -> np.ndarray:
    """Per-event 4th-root + sum normalisation (same as cluster fingerprint)."""
    V = np.maximum(snaps, 0.0) ** 0.25
    row_sums = V.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = np.finfo(float).eps
    return V / row_sums


class ClusterScatterWidget(QWidget):
    """2-D scatter of per-event amplitude fingerprints coloured by cluster label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._title = QLabel("EOD Identity Space — awaiting data")
        self._title.setStyleSheet("color:#abb2bf; font-size:8pt;")
        self._title.setAlignment(Qt.AlignCenter)
        root.addWidget(self._title)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#0d0d2a")
        root.addWidget(self._glw, stretch=1)

        self._plot = self._glw.addPlot()
        self._plot.setLabel("bottom", "Dim 1")
        self._plot.setLabel("left",   "Dim 2")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()

    def update_data(self, snaps: np.ndarray, labels: np.ndarray,
                    fname: str = "") -> None:
        """Project snaps to 2-D and scatter, coloured by cluster label.

        Args:
            snaps:  (N_events × N_ch) raw amplitude snapshots
            labels: (N_events,) DBSCAN labels (>= 0 = valid cluster, -1 = noise)
        """
        if len(snaps) < 4:
            return

        fp = _normalise_snaps(snaps)

        # Subsample for speed if very large
        if len(fp) > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(fp), 2000, replace=False)
            fp_fit, lbl_fit = fp[idx], labels[idx]
        else:
            fp_fit, lbl_fit = fp, labels

        # PCA only (UMAP would block the GUI thread for 30+ s on first JIT)
        try:
            from sklearn.decomposition import PCA
            coords_fit = PCA(n_components=2).fit_transform(fp_fit)
            method = "PCA"
        except Exception:
            return

        self._plot.clear()
        unique_labels = sorted(set(int(l) for l in lbl_fit))
        n_clusters = sum(1 for l in unique_labels if l >= 0)

        for lbl in unique_labels:
            mask = lbl_fit == lbl
            if lbl < 0:
                col = (130, 130, 130, 60)
                size = 3
            else:
                col = _rgba(lbl)
                size = 5
            scatter = pg.ScatterPlotItem(
                x=coords_fit[mask, 0].tolist(),
                y=coords_fit[mask, 1].tolist(),
                symbol='o', size=size,
                brush=pg.mkBrush(*col),
                pen=pg.mkPen(None),
            )
            self._plot.addItem(scatter)

        src = f"  [{fname}]" if fname else ""
        self._title.setText(
            f"EOD Identity ({method})  —  {len(snaps)} events  ·  {n_clusters} cluster(s){src}"
        )
