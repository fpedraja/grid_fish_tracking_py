"""Entry point for the Electric Fish Tracker GUI."""

import os
import sys

# Force pyqtgraph to use PyQt5 (must be set before any pyqtgraph import)
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Electric Fish Tracker")
    app.setStyle("Fusion")

    # pyqtgraph global defaults
    pg.setConfigOption("background", "#0d0d2a")
    pg.setConfigOption("foreground", "#e0e0e0")
    pg.setConfigOption("antialias", True)

    app.setFont(QFont("Segoe UI", 9))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
