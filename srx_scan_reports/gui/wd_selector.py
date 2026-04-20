


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QGridLayout,
    QPushButton,
    QLabel,
    QFrame,
    QSizePolicy,
    QLineEdit,
    QFileDialog
    )
from PyQt5.QtGui import QIntValidator




class WDWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.init_UI()

    
    def init_UI(self):

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        self.layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.wd_button = QPushButton('..', self)
        self.wd_button.clicked.connect(self.select_folder)
        self.wd_button.setFixedWidth(20)

        self.wd_line = QLineEdit('')
        self.wd_line.setReadOnly(True)

        self.layout.addWidget(self.wd_button)
        self.layout.addWidget(self.wd_line)

        self.setLayout(self.layout)


    def select_folder(self):
        if self.wd_line.text() != '':
            dir_current = self.wd_line.text()
        else:
            dir_current = '/nsls2/data/srx/proposals/'

        dir = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            dir_current,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if dir:
            self.wd_line.setText(dir)