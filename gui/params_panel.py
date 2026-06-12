"""Parameter controls panel – organised into collapsible QGroupBoxes."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QScrollArea, QLabel, QCheckBox, QComboBox,
)
from PyQt5.QtCore import Qt

from pipeline import FishTrackingConfig


def _dspin(val: float, lo: float, hi: float,
           step: float = 0.5, dec: int = 3) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    return w


def _ispin(val: int, lo: int, hi: int, step: int = 10) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    return w


class ParamsPanel(QScrollArea):
    """Scrollable panel containing all pipeline parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ---- Signal processing ----
        grp, form = self._group("Signal Processing")
        cfg = FishTrackingConfig()

        # ---- Notch filter ----
        ngrp, nform = self._group("Notch Filter")
        self.notch_enable = QCheckBox("Enable")
        self.notch_enable.setChecked(cfg.notch_enabled)
        self.notch_hz = _dspin(cfg.notch_hz, 40.0, 70.0, 10.0, 1)  # 50 Hz (Uruguay)
        self.notch_Q  = _dspin(cfg.notch_Q,   5.0, 200.0,  5.0, 1)
        nform.addRow("", self.notch_enable)
        nform.addRow("Frequency (Hz):", self.notch_hz)
        nform.addRow("Q factor:",       self.notch_Q)
        layout.addWidget(ngrp)

        self.bp_low   = _dspin(cfg.bp_low,  50,  5000, 50, 1)
        self.bp_high  = _dspin(cfg.bp_high, 100, 10000, 100, 1)
        self.bp_order = _ispin(cfg.bp_order, 1, 8, 1)
        form.addRow("BP low (Hz):",   self.bp_low)
        form.addRow("BP high (Hz):",  self.bp_high)
        form.addRow("Filter order:",  self.bp_order)
        layout.addWidget(grp)

        # ---- Peak detection ----
        grp, form = self._group("Peak Detection")
        self.min_pk_height = _dspin(cfg.min_pk_height, 1e-6, 10.0, 0.001, 5)
        self.mpd_ms        = _dspin(cfg.mpd_ms, 0.5, 50.0, 0.5, 1)
        self.merge_ms      = _dspin(cfg.merge_ms, 0.5, 20.0, 0.5, 1)
        form.addRow("Min peak height:", self.min_pk_height)
        form.addRow("Min peak dist (ms):", self.mpd_ms)
        form.addRow("Merge window (ms):", self.merge_ms)
        layout.addWidget(grp)

        # ---- Localization ----
        lgrp, lform = self._group("Localization")
        self.loc_method = QComboBox()
        self.loc_method.addItem("Gaussian grid (argmax)", "gaussian_grid")
        self.loc_method.addItem("Weighted centroid (Henninger 2020)", "weighted_centroid")
        self.top_n_electrodes = _ispin(cfg.top_n_electrodes, 1, 8, 1)
        lform.addRow("Method:", self.loc_method)
        lform.addRow("Top N electrodes:", self.top_n_electrodes)
        layout.addWidget(lgrp)

        # ---- Spatial ----
        grp, form = self._group("Spatial Grid")
        self.sigma_spatial = _dspin(cfg.sigma_spatial, 1.0, 100.0, 1.0, 1)
        form.addRow("Gaussian σ (cm):", self.sigma_spatial)
        layout.addWidget(grp)

        # ---- DBSCAN ----
        grp, form = self._group("DBSCAN Clustering")
        self.eps_phys = _dspin(cfg.eps_phys, 1.0, 100.0, 1.0, 1)
        self.min_pts  = _ispin(cfg.min_pts, 5, 5000, 10)
        form.addRow("Eps (cm):",    self.eps_phys)
        form.addRow("Min samples:", self.min_pts)
        layout.addWidget(grp)

        # ---- Cluster filters ----
        grp, form = self._group("Cluster Filters")
        self.min_events_for_fish    = _ispin(cfg.min_events_for_fish, 10, 100000, 50)
        self.min_events_per_cluster = _ispin(cfg.min_events_per_cluster, 10, 10000, 50)
        self.min_dur                = _dspin(cfg.min_dur_per_cluster, 0.5, 60.0, 0.5, 1)
        self.min_freq               = _dspin(cfg.min_freq_hz, 0.5, 500.0, 1.0, 1)
        self.max_freq               = _dspin(cfg.max_freq_hz, 1.0, 1000.0, 10.0, 1)
        form.addRow("Min events (file):",    self.min_events_for_fish)
        form.addRow("Min events (cluster):", self.min_events_per_cluster)
        form.addRow("Min duration (s):",     self.min_dur)
        form.addRow("Min freq (Hz):",        self.min_freq)
        form.addRow("Max freq (Hz):",        self.max_freq)
        layout.addWidget(grp)

        # ---- Tracker association gates ----
        grp, form = self._group("Tracker Association Gates")
        self.hard_gate_pos = _dspin(60.0,  1.0, 200.0, 5.0, 1)
        self.hard_gate_f   = _dspin(2.0,   0.1,  20.0, 0.5, 2)
        self.sig_gate      = _dspin(0.35,  0.0,   1.0, 0.05, 2)
        self.max_miss      = _ispin(10,    1,    100,   1)
        form.addRow("Hard gate pos (cm):",   self.hard_gate_pos)
        form.addRow("Hard gate freq (Hz):",  self.hard_gate_f)
        form.addRow("Amplitude gate:",       self.sig_gate)
        form.addRow("Max missed files:",     self.max_miss)
        layout.addWidget(grp)

        # ---- Tracker noise ----
        grp, form = self._group("Tracker Noise")
        self.meas_sig_pos  = _dspin(cfg.meas_sig_pos, 0.1, 50.0, 0.5, 1)
        self.meas_sig_f    = _dspin(cfg.meas_sig_f, 0.01, 5.0, 0.05, 3)
        self.wander_1min   = _dspin(cfg.wander_1min, 1.0, 200.0, 5.0, 1)
        self.freq_drift_1m = _dspin(cfg.freq_drift_1m, 0.01, 10.0, 0.05, 3)
        form.addRow("Pos noise σ (cm):", self.meas_sig_pos)
        form.addRow("Freq noise σ (Hz):", self.meas_sig_f)
        form.addRow("Wander/min (cm):",  self.wander_1min)
        form.addRow("Freq drift/min (Hz):", self.freq_drift_1m)
        layout.addWidget(grp)

        layout.addStretch()
        self.setWidget(inner)

    # ------------------------------------------------------------------
    def to_config(self) -> FishTrackingConfig:
        """Build a FishTrackingConfig from the current widget values."""
        cfg = FishTrackingConfig()
        cfg.bp_low               = self.bp_low.value()
        cfg.bp_high              = self.bp_high.value()
        cfg.bp_order             = self.bp_order.value()
        cfg.min_pk_height        = self.min_pk_height.value()
        cfg.mpd_ms               = self.mpd_ms.value()
        cfg.merge_ms             = self.merge_ms.value()
        cfg.localization_method  = self.loc_method.currentData()
        cfg.top_n_electrodes     = self.top_n_electrodes.value()
        cfg.sigma_spatial        = self.sigma_spatial.value()
        cfg.eps_phys             = self.eps_phys.value()
        cfg.min_pts              = self.min_pts.value()
        cfg.min_events_for_fish    = self.min_events_for_fish.value()
        cfg.min_events_per_cluster = self.min_events_per_cluster.value()
        cfg.min_dur_per_cluster    = self.min_dur.value()
        cfg.min_freq_hz          = self.min_freq.value()
        cfg.max_freq_hz          = self.max_freq.value()
        cfg.meas_sig_pos         = self.meas_sig_pos.value()
        cfg.meas_sig_f           = self.meas_sig_f.value()
        cfg.wander_1min          = self.wander_1min.value()
        cfg.freq_drift_1m        = self.freq_drift_1m.value()
        cfg.hard_gate_pos        = self.hard_gate_pos.value()
        cfg.hard_gate_f          = self.hard_gate_f.value()
        cfg.sig_gate             = self.sig_gate.value()
        cfg.max_miss             = self.max_miss.value()
        cfg.notch_enabled        = self.notch_enable.isChecked()
        cfg.notch_hz             = self.notch_hz.value()
        cfg.notch_Q              = self.notch_Q.value()
        return cfg

    # ------------------------------------------------------------------
    @staticmethod
    def _group(title: str) -> tuple:
        grp  = QGroupBox(title)
        form = QFormLayout(grp)
        form.setSpacing(4)
        form.setContentsMargins(8, 4, 8, 8)
        return grp, form
