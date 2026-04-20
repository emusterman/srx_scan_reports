


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
    QLineEdit
    )
from PyQt5.QtGui import QIntValidator

from .periodic_table import ToggleButton, QPeriodicTable, AspectRatioWidget



class ROIWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.defaults = {}
        self.enabled = {}

        self.init_UI()
        self._get_defaults()

        self.roi_controls.reset_button.clicked.connect(self.reset_defaults)

    
    def init_UI(self):

        label_style = '''QLabel { 
                            color: #555555; 
                            font-size: 14px;
                            font-weight : bold;
                        }
                         QPushButton {
                            color : #555555;
                            font-size : 14px;
                            font-weight : bold;
                         }'''

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.element_label = QLabel('Select Elements')
        self.element_label.setStyleSheet(label_style)
        self.periodic_table = AspectRatioWidget(QPeriodicTable(), 1.8)
        self.periodic_table.setMinimumSize(450, 250)
        layout.addWidget(self.element_label)
        layout.addWidget(self.periodic_table, 1)
        layout.addSpacing(20)
        layout.addStretch(0)

        
        # Scalers. Include or don't
        self.scalers_label = QLabel('Include Scalers')
        self.scalers_label.setStyleSheet(label_style)
        self.scaler_im_button = ToggleButton('im', states=[0, 1])
        self.scaler_i0_button = ToggleButton('i0', states=[0, 1])
        self.scaler_it_button = ToggleButton('it', states=[0, 1])
        self.scalers = [self.scaler_im_button,
                        self.scaler_i0_button,
                        self.scaler_it_button]
        for sc in self.scalers:
            sc.setFixedSize(50, 50)

        self.beamline = ArrowLine(self.scalers)

        scaler_box = QVBoxLayout()
        scaler_box.setContentsMargins(0, 0, 0, 0)
        scaler_box.setSpacing(5)
        scaler_box.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        scaler_box.addWidget(self.scalers_label)
        scaler_box.addWidget(self.beamline, 1)

        # Area Detectors. Exclude or don't
        self.area_detectors_label = QLabel('Exclude Area Detectors')
        self.area_detectors_label.setStyleSheet(label_style)
        self.merlin_button = ToggleButton('Merlin', states=[0, 3])
        self.dexela_button = ToggleButton('Dexela', states=[0, 3])
        self.eiger_button = ToggleButton('Eiger', states=[0, 3])
        self.area_detectors = [self.merlin_button,
                               self.dexela_button,
                               self.eiger_button]

        ad_box = QVBoxLayout()
        ad_box.setContentsMargins(0, 0, 0, 0)
        ad_box.setSpacing(5)
        ad_box.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        ad_box.addWidget(self.area_detectors_label)
        for ad in self.area_detectors:
            ad_box.addWidget(ad, 1, alignment=Qt.AlignVCenter | Qt.AlignHCenter)
            ad.setFixedSize(100, 50)
        
        ex_box = QHBoxLayout()
        ex_box.setContentsMargins(0, 0, 0, 0)
        ex_box.setSpacing(5)
        ex_box.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        ex_box.addLayout(scaler_box, 1)
        ex_box.addStretch(1)
        ex_box.addLayout(ad_box, 1)

        layout.addLayout(ex_box)

        # Add instructions
        self.roi_controls = ROIControls()

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(20)
        self.layout.addLayout(layout, 1)
        self.layout.addStretch(0)
        self.layout.addWidget(self.roi_controls, 0)

        self.setLayout(self.layout)
    

    def _get_defaults(self):

        for btn in (self.periodic_table.target.element_buttons + self.scalers + self.area_detectors):
            self.defaults[btn] = btn.state
            self.enabled[btn] = btn.isEnabled()
    

    def reset_defaults(self):

        for btn, state in self.defaults.items():
            btn.state = state
            btn.updateBackgroundColor()
            btn.update()
        
        self.periodic_table.target.selection_order = []



class ArrowLine(QWidget):
    def __init__(self, buttons, color="#888888", thickness=2):
        super().__init__()
        self.buttons = buttons
        self.color = color
        self.thickness = thickness
        
        self.init_UI()

    def init_UI(self):
        # 1. Setup the main horizontal layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)  # Keeps the line segments touching

        # 2. The "Tail" (Left-side line)
        tail = self._create_line_segment()
        layout.addWidget(tail, 1)  # Stretch=1 pushes everything to the right

        # 3. Add the Buttons with connecting segments
        for i, btn in enumerate(self.buttons):
            # Ensure buttons don't vertically stretch the whole line area
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            layout.addWidget(btn)

            # Add a small line segment AFTER the button (unless it's the last one)
            # or before the arrow head
            mid_segment = self._create_line_segment()
            mid_segment.setFixedWidth(20) # Fixed spacing between buttons/arrow
            layout.addWidget(mid_segment)

        # 4. The Arrow Head
        head = QLabel("▶")
        head.setStyleSheet(f"""
            color: {self.color}; 
            font-size: 20px; 
            background: transparent;
            margin-left: -2px; /* Pulls triangle tip onto the line */
        """)
        head.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        layout.addWidget(head)

    def _create_line_segment(self):
        """Helper to create a line fragment with the current theme."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"""
            background-color: {self.color}; 
            max-height: {self.thickness}px; 
            border: none;
        """)
        return line


class ROIControls(QWidget):

    def __init__(self):
        super().__init__()

        self.init_UI()
    
    
    def init_UI(self):

        label_style = '''QLabel { 
                            color: #555555; 
                            font-size: 14px;
                            font-weight : bold;
                        }
                         QPushButton {
                            color : #555555;
                            font-size : 14px;
                            font-weight : bold;
                         }'''

        # ROI order
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        self.layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.instruction_label = QLabel('ROI Selection Order')
        self.instruction_label.setStyleSheet(label_style)
        self.instruction_text = QLabel()
        self.instruction_text.setWordWrap(True)

        instructions = (
        "1. VLM images, if available"
        + "\n2. Scalers, if selected"
        + "\n3. Area detectors, if used and not excluded"
        + "\n4. Elements selected for specific scans"
        + "\n5. Elements selected for report"
        + "\n6. Elements found algorithmically")

        self.instruction_text.setText(instructions)
        self.instruction_text.setFixedWidth(275)

        self.layout.addStretch(1)
        self.layout.addWidget(self.instruction_label)
        self.layout.addWidget(self.instruction_text)
        self.layout.addSpacing(10)

        # Color key
        self.color_key_label = QLabel('Color Key')
        self.color_key_label.setStyleSheet(label_style)
        self.layout.addWidget(self.color_key_label)

        for color, label in zip(ToggleButton('')._colors.values(), ['Possible', 'Include', 'Ignore', 'Exclude']):
            color_square = QLabel()
            color_square.setFixedSize(20, 20)
            color_square.setStyleSheet(f"""
                background-color: {color.name()};
                border: 1px solid #000000;
                border-radius: 3px;
            """)
            color_label = QLabel(label)

            hbox = QHBoxLayout()
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            hbox.setSpacing(5)

            hbox.addWidget(color_square)
            hbox.addWidget(color_label)
            hbox.addStretch(1)

            setattr(self, f'{label}_color_square', color_square)
            setattr(self, f'{label}_color_label', color_label)

            self.layout.addLayout(hbox)
        self.layout.addSpacing(10)


        # ROI number
        self.roi_num_label = QLabel('Number of ROIs')
        self.roi_num_label.setStyleSheet(label_style)
        self.min_roi_num_label = QLabel('Minimum')
        self.min_roi_num = QLineEdit('1')
        self.min_roi_num.setValidator(QIntValidator(1, 10))
        # self.min_roi_num.setFixedWidth(275)

        self.max_roi_num_label = QLabel('Maximum')
        self.max_roi_num = QLineEdit('10')
        self.max_roi_num.setValidator(QIntValidator(1, 10))
        # self.max_roi_num.setFixedWidth(275)

        self.layout.addWidget(self.roi_num_label)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.addWidget(self.min_roi_num_label, 0, 0)
        grid.addWidget(self.min_roi_num, 1, 0)
        grid.addWidget(self.max_roi_num_label, 0, 1)
        grid.addWidget(self.max_roi_num, 1, 1)
        self.layout.addLayout(grid)
        self.layout.addSpacing(10)


        # Reset
        self.reset_button = QPushButton('Reset ROIs')
        self.reset_button.setStyleSheet(label_style)
        # self.reset_button.setFixedWidth(275)

        self.layout.addWidget(self.reset_button)
        self.layout.addStretch(1)


        self.setLayout(self.layout)