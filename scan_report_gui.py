

import sys
import argparse
from IPython import get_ipython

from PyQt5.QtWidgets import (
    QApplication
)

print('Importing SRX Scan Report GUI modules...')
from srx_scan_reports.gui.main_window import MainWindow

def run_gui():

    # Start application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    window.resize(900, 1200)
    sys.exit(app.exec_())


if __name__ == "__main__":
    print('Starting SRX Scan Report GUI...')
    run_gui()