

__version__ = "1.0.0"

import sys

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PyQt5.QtGui import QFont

from .central_widget import CentralWidget




_main_window_geometry = {
    "initial_height": 700,
    "initial_width": 1200,
    "min_height": 700,
    "min_width": 1200,
}



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.init_UI()

    
    def init_UI(self):
        
        # Set window default sizes
        self.resize(_main_window_geometry["initial_width"],
                    _main_window_geometry["initial_height"])
        self.setMinimumWidth(_main_window_geometry["min_width"])
        self.setMinimumHeight(_main_window_geometry["min_height"])

        # Set window title and icon
        self.setWindowTitle(f'SRX Scan Report GUI v{__version__}')
        # file_path = os.path.dirname(os.path.dirname(__file__))
        # icon_path = os.path.join(file_path, 'data/srx_logo.png')
        # self.setWindowIcon(QIcon(icon_path))

        # Set central widget
        self.central_widget = CentralWidget()
        self.setCentralWidget(self.central_widget)

        # Setup status bar to cheaply catch terminal outputs
        self.status_display = TerminalLabel(max_lines=3)
        self.statusBar().addWidget(self.status_display, 1)
        self.statusBar().setMinimumHeight(22 * self.status_display.max_lines)

        self.statusBar().setStyleSheet("""
            QStatusBar {
                border-top: 2px solid #888888;
            }
        """)

        self.redirector = StatusBarRedirector()
        self.redirector.message_received.connect(self.status_display.update_text)
        sys.stdout = self.redirector
        print('Ready')


    def closeEvent(self, event):
        # Restore normal terminal output when closing
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


class TerminalLabel(QWidget):
    def __init__(self, max_lines=3, parent=None):
        super().__init__(parent)
        self.max_lines = max_lines
        self.history = []  # Standard list for history
        self._pending_overwrite = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(2)
        
        # Create a list of labels
        self.labels = []
        for _ in range(max_lines):
            lbl = QLabel("")
            lbl.setStyleSheet("font-family: Arial; font-size: 14px; color: black;")
            layout.addWidget(lbl)
            self.labels.append(lbl)

    def update_text(self, text):

        # Check if text was supposed to be an overwrite of last text
        if text == '\r':
            self._pending_overwrite = True
            return

        clean_text = text.strip()
        if not clean_text:
            return
        
        if self._pending_overwrite:
            self.history[-1] = clean_text
            self._pending_overwrite = False
        else:
            self.history.append(clean_text)
        
        # If list exceeds max_lines, remove the oldest (first) item
        if len(self.history) > self.max_lines:
            self.history.pop(0)
        
        # Clear all labels first (in case history is shorter than max_lines)
        for lbl in self.labels:
            lbl.setText("")
            
        # Fill labels from the bottom up
        # Iterate backwards through history and labels
        for i, msg in enumerate(reversed(self.history)):
            # Index from the end of the labels list
            self.labels[-(i + 1)].setText(msg)

    # def update_text(self, text):

    #     # Check if text was supposed to be an overwrite of last text
    #     if text[-2:] == '\r':
    #         self.history[-1] = text.strip()
    #     else:
    #         self.history.append(text.strip())
        
    #     # If list exceeds max_lines, remove the oldest (first) item
    #     if len(self.history) > self.max_lines:
    #         self.history.pop(0)
        
    #     # Clear all labels first (in case history is shorter than max_lines)
    #     for lbl in self.labels:
    #         lbl.setText("")
            
    #     # Fill labels from the bottom up
    #     # Iterate backwards through history and labels
    #     for i, msg in enumerate(reversed(self.history)):
    #         # Index from the end of the labels list
    #         self.labels[-(i + 1)].setText(msg)
        

class StatusBarRedirector(QObject):
    # Signal to carry the text to the status bar safely
    message_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.terminal = sys.__stdout__

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()

        # Send signal with message to gui
        self.message_received.emit(message)

    def flush(self):
        self.terminal.flush()
        pass