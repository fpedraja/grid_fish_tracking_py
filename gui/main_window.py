"""Main application window."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QProgressBar, QTabWidget, QTextEdit, QSplitter,
    QMessageBox, QStatusBar, QMenuBar, QAction, QSlider,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

from gui.params_panel import ParamsPanel
from gui.plot_widget import TrackPlotWidget, FreqPlotWidget
from gui.results_table import ResultsTable
from gui.signal_viewer import SignalViewer
from gui.umap_widget import ClusterScatterWidget
from worker import ProcessingWorker
from loader_worker import SignalLoaderWorker
from pipeline import FileResult, FishTrackingConfig


_DARK_TEXT  = "#e0e0e0"
_ACCENT     = "#61afef"
_BTN_START  = "#98c379"
_BTN_STOP   = "#e06c75"
_BTN_EXPORT = "#c678dd"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electric Fish Tracker")
        self.resize(1300, 800)

        self._worker:        ProcessingWorker | None = None
        self._loader:        SignalLoaderWorker | None = None
        self._results:       list[FileResult] = []
        self._df:            pd.DataFrame | None = None
        self._folder:        str = ""
        self._file_list:     list[str] = []      # ordered wav filenames
        self._file_idx_map:  dict[str, int] = {} # fname → file_idx (1-based)
        self._last_config:   FishTrackingConfig | None = None

        self._build_menu()
        self._build_ui()
        self._apply_stylesheet()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = QMenuBar(self)
        self.setMenuBar(mb)
        file_menu = mb.addMenu("File")

        act_open = QAction("Open Folder…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._browse_folder)
        file_menu.addAction(act_open)

        act_export = QAction("Export CSV…", self)
        act_export.setShortcut("Ctrl+S")
        act_export.triggered.connect(self._export_csv)
        file_menu.addAction(act_export)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- Top bar: folder path + buttons ----
        top = QHBoxLayout()
        top.setSpacing(6)

        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select WAV folder…")
        self._folder_edit.setReadOnly(True)

        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse_folder)

        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setFixedWidth(100)
        self._btn_start.clicked.connect(self._start_processing)
        self._btn_start.setEnabled(False)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedWidth(100)
        self._btn_stop.clicked.connect(self._stop_processing)
        self._btn_stop.setEnabled(False)

        self._btn_export = QPushButton("⬇  Export CSV")
        self._btn_export.setFixedWidth(120)
        self._btn_export.clicked.connect(self._export_csv)
        self._btn_export.setEnabled(False)

        self._btn_clear = QPushButton("✕  Clear")
        self._btn_clear.setFixedWidth(80)
        self._btn_clear.clicked.connect(self._clear_results)

        top.addWidget(QLabel("Folder:"))
        top.addWidget(self._folder_edit, stretch=1)
        top.addWidget(btn_browse)
        top.addWidget(self._btn_start)
        top.addWidget(self._btn_stop)
        top.addWidget(self._btn_export)
        top.addWidget(self._btn_clear)
        root.addLayout(top)

        # ---- Progress bar ----
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m files  (%p%)")
        self._progress.setValue(0)
        root.addWidget(self._progress)

        # ---- Main splitter: params | content ----
        splitter = QSplitter(Qt.Horizontal)

        self._params = ParamsPanel()
        splitter.addWidget(self._params)

        # Right side: tabs
        self._tabs = QTabWidget()

        # Tab 0 – Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._tabs.addTab(self._log, "Log")

        # Tab 1 – Results table
        self._results_table = ResultsTable()
        self._tabs.addTab(self._results_table, "Results")

        # Tab 2 – Combined: track plot (left) + signal viewer (right)
        combined = QWidget()
        combined_layout = QVBoxLayout(combined)
        combined_layout.setContentsMargins(0, 0, 0, 0)
        combined_layout.setSpacing(2)

        viz_splitter = QSplitter(Qt.Horizontal)

        # --- left pane: track plot + freq plot + file slider ---
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 4)
        left_layout.setSpacing(3)

        left_vsplit = QSplitter(Qt.Vertical)
        self._plot = TrackPlotWidget(xlim=(0, 80), ylim=(0, 120))
        self._freq_plot = FreqPlotWidget()
        left_vsplit.addWidget(self._plot)
        left_vsplit.addWidget(self._freq_plot)
        left_vsplit.setSizes([320, 180])
        left_layout.addWidget(left_vsplit, stretch=1)

        # File / minute slider
        fslider_row = QHBoxLayout()
        fslider_row.addWidget(QLabel("File:"))
        self._file_pos_label = QLabel("—")
        self._file_pos_label.setFixedWidth(44)
        self._file_pos_label.setStyleSheet("color:#e5c07b; font-family:Consolas;")
        self._file_slider = QSlider(Qt.Horizontal)
        self._file_slider.setMinimum(0)
        self._file_slider.setMaximum(0)
        self._file_slider.setValue(0)
        self._file_slider.setEnabled(False)
        self._file_slider.valueChanged.connect(self._on_file_slider)
        self._file_name_label = QLabel("")
        self._file_name_label.setStyleSheet("color:#abb2bf; font-size:8pt;")
        fslider_row.addWidget(self._file_pos_label)
        fslider_row.addWidget(self._file_slider, stretch=1)
        fslider_row.addWidget(self._file_name_label)
        left_layout.addLayout(fslider_row)

        viz_splitter.addWidget(left_pane)

        # --- right pane: signal viewer (top) + UMAP scatter (bottom) ---
        right_pane = QSplitter(Qt.Vertical)

        self._signal_viewer = SignalViewer()
        right_pane.addWidget(self._signal_viewer)

        self._umap_widget = ClusterScatterWidget()
        right_pane.addWidget(self._umap_widget)
        right_pane.setSizes([400, 300])

        viz_splitter.addWidget(right_pane)
        viz_splitter.setSizes([440, 760])

        combined_layout.addWidget(viz_splitter)
        self._tabs.addTab(combined, "Tracks & Signals")

        splitter.addWidget(self._tabs)
        splitter.setSizes([280, 1020])
        root.addWidget(splitter, stretch=1)

        # ---- Status bar ----
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("Ready")
        self._status.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Folder selection
    # ------------------------------------------------------------------

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select WAV folder",
                                                  self._folder or "")
        if folder:
            self._folder = folder
            self._folder_edit.setText(folder)
            self._btn_start.setEnabled(True)
            self._log_msg(f"Folder: {folder}")
            n = len([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
            self._status_label.setText(f"{n} WAV files found")
            # Push sensor positions to the plot
            cfg = self._params.to_config()
            self._plot.set_sensors(cfg.xy_meas)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _start_processing(self) -> None:
        if not self._folder:
            QMessageBox.warning(self, "No folder", "Please select a WAV folder first.")
            return

        cfg = self._params.to_config()
        self._last_config = cfg

        # Count files to set progress max
        wav_files = sorted(f for f in os.listdir(self._folder) if f.lower().endswith(".wav"))
        self._file_list = wav_files
        self._file_idx_map = {f: i for i, f in enumerate(wav_files)}
        self._progress.setMaximum(len(wav_files))
        self._progress.setValue(0)

        self._worker = ProcessingWorker(self._folder, cfg, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.signal_ready.connect(self._on_signal_ready)
        self._worker.cluster_ready.connect(self._on_cluster_ready)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._params.setEnabled(False)
        self._status_label.setText("Processing…")
        self._log_msg("=== Processing started ===")

        self._worker.start()

    def _stop_processing(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._log_msg("Stop requested – waiting for current file…")
            self._btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    # Worker signals
    # ------------------------------------------------------------------

    def _on_progress(self, file_idx: int, total: int,
                     fname: str, msg) -> None:
        self._progress.setValue(file_idx + 1)
        if msg:
            self._log_msg(f"  [{file_idx+1}/{total}] {fname}: {msg}", color="#e5c07b")
        else:
            self._log_msg(f"[{file_idx+1}/{total}] {fname}")

    def _on_file_done(self, result: FileResult) -> None:
        self._results.append(result)
        self._results_table.add_result(result)
        file_idx = self._file_idx_map.get(result.filename, len(self._results) - 1)
        self._plot.record(file_idx, result)
        self._freq_plot.record(file_idx, result)
        # Update file slider range and move to latest
        self._file_slider.setMaximum(self._plot.max_file_idx)
        self._file_slider.setEnabled(True)
        self._file_slider.setValue(file_idx)
        # Switch to combined tab on first result
        if len(self._results) == 1:
            self._tabs.setCurrentIndex(2)

    def _on_signal_ready(self, fs: float, snippet, event_times_sec, labels, fname: str) -> None:
        self._signal_viewer.set_data(fs, snippet, event_times_sec, labels, fname)

    def _on_cluster_ready(self, snaps, labels, fname: str) -> None:
        self._umap_widget.update_data(snaps, labels, fname)

    def _on_file_slider(self, val: int) -> None:
        """Scrub the track plot and signal viewer to file index val."""
        self._plot.show_up_to(val)
        self._freq_plot.show_up_to(val)
        # Update label
        if val < len(self._file_list):
            fname = self._file_list[val]
            self._file_pos_label.setText(f"{val + 1}")
            self._file_name_label.setText(fname)
        else:
            fname = ""
            self._file_pos_label.setText(str(val + 1))

        # Load signals for this file in background
        if fname and self._folder and self._last_config:
            fpath = os.path.join(self._folder, fname)
            if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                if self._loader and self._loader.isRunning():
                    self._loader.terminate()
                    self._loader.wait()
                self._loader = SignalLoaderWorker(fpath, fname,
                                                  self._last_config, parent=self)
                self._loader.signal_ready.connect(self._on_signal_ready)
                self._loader.error.connect(lambda e: self._log_msg(f"Loader: {e}", color=_BTN_STOP))
                self._loader.start()

    def _on_finished(self, df: pd.DataFrame) -> None:
        self._df = df
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_export.setEnabled(not df.empty)
        self._params.setEnabled(True)
        n_fish = df["track_id"].nunique() if not df.empty else 0
        msg = (f"Done – {len(self._results)} files processed, "
               f"{n_fish} unique fish tracked.")
        self._log_msg(f"=== {msg} ===", color=_BTN_START)
        self._status_label.setText(msg)

    def _on_error(self, tb: str) -> None:
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._params.setEnabled(True)
        self._log_msg(f"ERROR:\n{tb}", color=_BTN_STOP)
        self._status_label.setText("Error – see Log tab")
        QMessageBox.critical(self, "Processing error", tb[:500])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        if self._df is None or self._df.empty:
            QMessageBox.information(self, "Nothing to export",
                                    "Run processing first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", self._folder or "",
            "CSV files (*.csv)"
        )
        if path:
            self._df.to_csv(path, index=False)
            self._log_msg(f"Exported → {path}", color=_BTN_EXPORT)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def _clear_results(self) -> None:
        self._results.clear()
        self._df = None
        self._results_table.clear_results()
        self._plot.clear_all()
        self._freq_plot.clear_all()
        self._log.clear()
        self._progress.setValue(0)
        self._btn_export.setEnabled(False)
        self._file_slider.setMaximum(0)
        self._file_slider.setValue(0)
        self._file_slider.setEnabled(False)
        self._file_pos_label.setText("—")
        self._file_name_label.setText("")
        self._status_label.setText("Cleared")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_msg(self, text: str, color: str = _DARK_TEXT) -> None:
        self._log.append(f'<span style="color:{color};">{text}</span>')

    # ------------------------------------------------------------------
    # Stylesheet (dark theme – matches existing project)
    # ------------------------------------------------------------------

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d0d2a;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #2a2a5a;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
                color: #abb2bf;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit {
                background-color: #1a1a3a;
                border: 1px solid #2a2a5a;
                border-radius: 3px;
                color: #e0e0e0;
                padding: 2px 4px;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #61afef;
            }
            QPushButton {
                background-color: #1e2a3a;
                border: 1px solid #2a4a6a;
                border-radius: 4px;
                color: #e0e0e0;
                padding: 4px 10px;
            }
            QPushButton:hover  { background-color: #2a3a5a; }
            QPushButton:pressed { background-color: #1a2a4a; }
            QPushButton:disabled { color: #555577; border-color: #222244; }
            QProgressBar {
                background-color: #1a1a3a;
                border: 1px solid #2a2a5a;
                border-radius: 3px;
                height: 16px;
                text-align: center;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #61afef;
                border-radius: 2px;
            }
            QTabWidget::pane {
                border: 1px solid #2a2a5a;
                background-color: #0d0d2a;
            }
            QTabBar::tab {
                background-color: #1a1a3a;
                border: 1px solid #2a2a5a;
                border-bottom: none;
                padding: 4px 16px;
                color: #abb2bf;
            }
            QTabBar::tab:selected {
                background-color: #0d0d2a;
                color: #61afef;
                border-top: 2px solid #61afef;
            }
            QTableWidget {
                background-color: #0d0d2a;
                gridline-color: #1e1e3e;
                alternate-background-color: #111128;
                color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #1a1a3a;
                border: 1px solid #2a2a5a;
                padding: 3px 6px;
                color: #abb2bf;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background: #1a1a3a;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #2a2a5a;
                min-height: 20px;
                border-radius: 5px;
            }
            QMenuBar {
                background-color: #111128;
                color: #e0e0e0;
            }
            QMenuBar::item:selected { background-color: #1e2a3a; }
            QMenu {
                background-color: #1a1a3a;
                border: 1px solid #2a2a5a;
                color: #e0e0e0;
            }
            QMenu::item:selected { background-color: #2a3a5a; }
            QStatusBar {
                background-color: #111128;
                color: #abb2bf;
            }
            QSplitter::handle { background-color: #2a2a5a; }
        """)
        # Colour the start/stop/export buttons distinctively
        self._btn_start.setStyleSheet(
            f"QPushButton {{ color: {_BTN_START}; border-color: {_BTN_START}; }}"
            f"QPushButton:hover {{ background-color: #1e3a2a; }}"
        )
        self._btn_stop.setStyleSheet(
            f"QPushButton {{ color: {_BTN_STOP}; border-color: {_BTN_STOP}; }}"
            f"QPushButton:hover {{ background-color: #3a1e1e; }}"
        )
        self._btn_export.setStyleSheet(
            f"QPushButton {{ color: {_BTN_EXPORT}; border-color: {_BTN_EXPORT}; }}"
            f"QPushButton:hover {{ background-color: #2a1e3a; }}"
        )
