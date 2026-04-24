"""Summary results table widget."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from pipeline import FileResult


_HEADERS = ["File", "Events", "Clusters", "Fish IDs", "Frequencies (Hz)", "Positions (cm)", "Std x,y (cm)"]


class ResultsTable(QTableWidget):
    """One row per processed file showing key tracking summary."""

    def __init__(self, parent=None):
        super().__init__(0, len(_HEADERS), parent)
        self.setHorizontalHeaderLabels(_HEADERS)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setDefaultSectionSize(22)

    def add_result(self, result: FileResult) -> None:
        row = self.rowCount()
        self.insertRow(row)

        ids   = ", ".join(str(f["id"])  for f in result.fish)
        freqs = ", ".join(f'{f["freq"]:.1f}' for f in result.fish)
        pos   = "  |  ".join(
            f'#{f["id"]} ({f["x"]:.0f},{f["y"]:.0f})' for f in result.fish
        )
        stds  = "  |  ".join(
            f'#{f["id"]} ({f.get("std_x", 0):.1f},{f.get("std_y", 0):.1f})'
            for f in result.fish
        )

        values = [
            result.filename,
            str(result.n_events),
            str(result.n_clusters),
            ids   or "—",
            freqs or "—",
            pos   or "—",
            stds  or "—",
        ]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.setItem(row, col, item)

        self.scrollToBottom()

    def clear_results(self) -> None:
        self.setRowCount(0)
