

__version__ = "0.0.1"


import os

from PyQt5.QtCore import (
    Qt,
    QObject,
    pyqtSignal,
    QThread
)
from PyQt5.QtWidgets import (
    QWidget,
    QLayout,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QCheckBox,
    QPushButton,
    QComboBox,
    QSizePolicy,
    QFrame
)
from PyQt5.QtGui import (
    QIntValidator
)


from .roi_selector import ROIWidget
from .wd_selector import WDWidget
from ..core import generate_scan_report


_main_window_geometry = {
    "initial_height": 1200,
    "initial_width": 900,
    "min_height": 1200,
    "min_width": 900,
}



class CentralWidget(QWidget):

    def __init__(self):
        super().__init__()

        self._is_running = False
        self._is_paused = False
        self.worker = None

        self.root_path = '/nsls2/data/srx/proposals/'

        self.init_UI()

        self.cycle.currentTextChanged.connect(self.get_available_proposals)


    def init_UI(self):

        title_style = '''QLabel { 
                            color: black; 
                            font-size: 16px;
                            font-weight : bold;
                        }'''
        
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setSizeConstraint(QLayout.SetMinimumSize)

        scan_range_title = QLabel('Scan Range Controls')
        scan_range_title.setStyleSheet(title_style)
        scan_range = self.add_scan_range()
        self.layout.addWidget(scan_range_title)
        self.layout.addLayout(scan_range)

        self.layout.addWidget(self.get_separator('horizontal', 3), 0)
        self.layout.addSpacing(10)

        select_rois_title = QLabel('ROI Selection Controls')
        select_rois_title.setStyleSheet(title_style)
        self.roi_selector = ROIWidget()
        self.layout.addWidget(select_rois_title)
        self.layout.addWidget(self.roi_selector, 1)

        self.layout.addWidget(self.get_separator('horizontal', 3), 0)
        self.layout.addStretch(0)

        run_title = QLabel('Run Controls')
        run_title.setStyleSheet(title_style)
        control_buttons = self.add_control_buttons()
        self.layout.addWidget(run_title)
        self.layout.addLayout(control_buttons)

        self.setLayout(self.layout)


    def add_scan_range(self):

        label_style = '''QLabel, QPushButton { 
                            color: #555555; 
                            font-size: 14px;
                            font-weight : bold;
                        }
                         QPushButton:disabled {
                            color: #aaaaaa;
                            font-weight: normal;
                         }'''

        def labeled_layout(label, line):
            vbox = QVBoxLayout()
            vbox.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(5)
            vbox.addWidget(label)
            vbox.addWidget(line)

            return vbox
        
        hbox = QHBoxLayout()
        hbox.setContentsMargins(10, 10, 10, 10)
        hbox.setSpacing(10)
        hbox.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.start_label = QLabel('Start ID')
        self.start_label.setStyleSheet(label_style)
        self.start_id = QLineEdit()
        self.start_id.setValidator(QIntValidator(int(-1e2), int(1e6)))

        self.end_label = QLabel('End ID')
        self.end_label.setStyleSheet(label_style)
        self.end_id = QLineEdit()
        self.end_id.setValidator(QIntValidator(int(-1e2), int(1e6)))

        id_layout = QVBoxLayout()
        id_layout.setContentsMargins(0, 0, 0, 0)
        id_layout.addLayout(labeled_layout(self.start_label, self.start_id))
        id_layout.addLayout(labeled_layout(self.end_label, self.end_id))

        self.cycle_label = QLabel('Cycle')
        self.cycle_label.setStyleSheet(label_style)
        self.cycle = QComboBox()
        self.cycle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.get_available_cycles()     

        self.proposal_label = QLabel('Proposal ID')
        self.proposal_label.setStyleSheet(label_style)
        self.proposal_id = QComboBox()
        self.proposal_id.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        pc_layout = QVBoxLayout()
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.addLayout(labeled_layout(self.cycle_label, self.cycle))
        pc_layout.addLayout(labeled_layout(self.proposal_label, self.proposal_id))

        self.wd_label = QLabel('Working Directory')
        self.wd_label.setStyleSheet(label_style)
        self.wd = WDWidget()

        self.continuous_label = QLabel('Continuous')
        self.continuous_label.setStyleSheet(label_style)
        self.continuous = QCheckBox('')
        self.continuous.setChecked(True)
        self.continuous.setStyleSheet('''QCheckBox::indicator {width: 20px; height: 20px;}''')

        self.verbose_label = QLabel('Verbose')
        self.verbose_label.setStyleSheet(label_style)
        self.verbose = QCheckBox('')
        self.verbose.setChecked(False)
        self.verbose.setStyleSheet('''QCheckBox::indicator {width: 20px; height: 20px;}''')    

        self.clear_button = QPushButton('Clear')
        self.clear_button.setStyleSheet(label_style)
        self.clear_button.clicked.connect(self.clear)

        extra_layout = QVBoxLayout()
        extra_layout.setContentsMargins(0, 0, 0, 0)
        extra_layout.addLayout(labeled_layout(self.wd_label, self.wd))

        other_layout = QHBoxLayout()
        other_layout.setContentsMargins(0, 0, 0, 0)

        cont_layout = QVBoxLayout()
        cont_layout.setContentsMargins(0, 0, 0, 0)
        cont_layout.setSpacing(5)
        cont_layout.addWidget(self.continuous_label, alignment=Qt.AlignVCenter | Qt.AlignHCenter)
        cont_layout.addWidget(self.continuous, alignment=Qt.AlignVCenter | Qt.AlignHCenter)
        other_layout.addLayout(cont_layout)

        verb_layout = QVBoxLayout()
        verb_layout.setContentsMargins(0, 0, 0, 0)
        verb_layout.setSpacing(5)
        verb_layout.addWidget(self.verbose_label, alignment=Qt.AlignVCenter | Qt.AlignHCenter)
        verb_layout.addWidget(self.verbose, alignment=Qt.AlignVCenter | Qt.AlignHCenter)
        other_layout.addLayout(verb_layout)

        other_layout.addWidget(self.clear_button, alignment=Qt.AlignBottom | Qt.AlignHCenter)
        extra_layout.addLayout(other_layout)

        hbox.addLayout(id_layout, 1)
        hbox.addWidget(self.get_separator('vertical', 2), 0)
        hbox.addLayout(pc_layout, 1)
        hbox.addWidget(self.get_separator('vertical', 2), 0)
        # hbox.addStretch(0)
        hbox.addLayout(extra_layout, 1)

        return hbox

    
    def add_control_buttons(self):

        self.setStyleSheet('''
            /* Base Style for all PushButtons */
            QPushButton[btnClass] {
                font-size: 20px;
                font-weight: bold;
                border-radius: 12px;
                border: 1px solid rgba(0, 0, 0, 40); 
                padding: 10px;
                color: #000000;
            }

            /* Base Disabled Style (Grayer text and flat border) */
            QPushButton[btnClass]:disabled {
                color: #888888; /* Faded text */
                border-bottom: 1px solid rgba(0, 0, 0, 20); /* Remove the 3D depth */
            }

            /* --- RUN BUTTON (Green) --- */
            QPushButton[btnClass="run"] {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d1f7d1, stop:1 #90ee90);
                border-bottom: 3px solid #76c776;
            }
            QPushButton[btnClass="run"]:hover { background-color: #a5f5a5; }
            QPushButton[btnClass="run"]:pressed { background-color: #7ccd7c; border-bottom: 1px solid #76c776; }
            /* Desaturated Green */
            QPushButton[btnClass="run"]:disabled {
                background-color: #d8e3d8; /* Muted grey-green */
            }

            /* --- PAUSE BUTTON (Cream/Tan) --- */
            QPushButton[btnClass="pause"] {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f9f5da, stop:1 #e1d7a0);
                border-bottom: 3px solid #b4aa78;
            }
            QPushButton[btnClass="pause"]:hover { background-color: #eee6b4; }
            QPushButton[btnClass="pause"]:pressed { background-color: #c8be8c; border-bottom: 1px solid #b4aa78; }
            /* Desaturated Cream */
            QPushButton[btnClass="pause"]:disabled {
                background-color: #e5e2d5; /* Muted grey-tan */
            }

            /* --- STOP BUTTON (Red) --- */
            QPushButton[btnClass="stop"] {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffb3b3, stop:1 #ff8080);
                border-bottom: 3px solid #dc6464;
            }
            QPushButton[btnClass="stop"]:hover { background-color: #ff9999; }
            QPushButton[btnClass="stop"]:pressed { background-color: #e67373; border-bottom: 1px solid #dc6464; }
            /* Desaturated Red */
            QPushButton[btnClass="stop"]:disabled {
                background-color: #e8d5d5; /* Muted grey-red */
            }
        ''')
        
        # Bottom Buttons
        hbox = QHBoxLayout()
        hbox.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        hbox.setContentsMargins(10, 10, 10, 10)

        self.run_button = QPushButton('Run')
        self.run_button.setProperty("btnClass", "run")
        self.run_button.setFixedSize(200, 100)
        self.run_button.clicked.connect(self.run_scan_report)

        # Create Pause Button
        self.pause_button = QPushButton('Pause')
        self.pause_button.setProperty("btnClass", "pause")
        self.pause_button.setFixedSize(200, 100)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_scan_report)

        # Create Stop Button
        self.stop_button = QPushButton('Stop')
        self.stop_button.setProperty("btnClass", "stop")
        self.stop_button.setFixedSize(200, 100)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_scan_report)

        hbox.addWidget(self.run_button)
        hbox.addStretch(1)
        hbox.addWidget(self.pause_button)
        hbox.addStretch(1)
        hbox.addWidget(self.stop_button)

        return hbox

    
    def get_separator(self, orientation='horizontal', thickness=3):
        line = QFrame()
        if orientation == "horizontal":
            line.setFrameShape(QFrame.HLine)
            line.setStyleSheet(f"""
                                background-color: #bbbbbb; 
                                max-height: {thickness}px; 
                                border: none;
                                margin: 10px 0px;
                                """)

        else:
            line.setFrameShape(QFrame.VLine)
            line.setStyleSheet(f"""
                                background-color: #bbbbbb; 
                                max-width: {thickness}px; 
                                border: none;
                                margin: 0px 10px;
                                """)
        line.setFrameShadow(QFrame.Sunken)

        return line

    
    def get_available_cycles(self, *args, **kwargs):
        self.cycle.clear()
        self.cycle.addItem('')

        cycles = os.listdir(self.root_path)
        cycles = sorted(cycles, reverse=True)
        cycles.remove('commissioning')
        for cycle in cycles:
            self.cycle.addItem(cycle)
        # Commissioning should go last
        self.cycle.addItem('commissioning')
    

    def get_available_proposals(self, cycle):
        self.proposal_id.clear()
        self.proposal_id.addItem('')

        proposals = [subdir.split('-')[-1] for subdir
                     in os.listdir(f'{self.root_path}{cycle}')]
        proposals = sorted(proposals)
        for proposal in proposals:
            self.proposal_id.addItem(proposal)
    

    def clear(self):

        self.start_id.setText('')
        self.end_id.setText('')
        self.continuous.setChecked(True)
        self.verbose.setChecked(False)
        self.wd.wd_line.setText('')
        self.proposal_id.setCurrentText('')
        self.cycle.setCurrentText('')

    
    # Pseudo-toggle
    def enable_scan_inputs(self, boolean):

        self.start_id.setEnabled(boolean)
        self.end_id.setEnabled(boolean)
        self.wd.wd_line.setEnabled(boolean)
        self.wd.wd_button.setEnabled(boolean)
        self.continuous.setEnabled(boolean)
        self.verbose.setEnabled(boolean)
        self.proposal_id.setEnabled(boolean)
        self.cycle.setEnabled(boolean)
        self.clear_button.setEnabled(boolean)

    
    # Pseudo-toggle
    def enable_roi_inputs(self, boolean):

        # ROI buttons
        for btn, value in self.roi_selector.enabled.items():
            if boolean is False:
                btn.setEnabled(boolean)
            else:
                btn.setEnabled(value)
        
        self.roi_selector.roi_controls.min_roi_num.setEnabled(boolean)
        self.roi_selector.roi_controls.max_roi_num.setEnabled(boolean)
        self.roi_selector.roi_controls.reset_button.setEnabled(boolean)


    def run_scan_report(self):
        
        # Set controls states
        self._set_running_state()

        self.params = {}

        # Get scan range arguments
        start_id = self.start_id.text()
        if start_id == '':
            self.params['start_id'] = None
        else:
            self.params['start_id'] = int(start_id)

        end_id = self.end_id.text()
        if end_id == '':
            self.params['end_id'] = None
        else:
            self.params['end_id'] = int(end_id)

        cycle = self.cycle.currentText()
        if cycle == '':
            self.params['cycle'] = None
        else:
            self.params['cycle'] = cycle            

        proposal_id = self.proposal_id.currentText()
        if proposal_id == '':
            self.params['proposal_id'] = None
        else:
            self.params['proposal_id'] = int(proposal_id)

        wd = self.wd.wd_line.text()
        if wd == '':
            self.params['wd'] = None
        else:
            self.params['wd'] = wd

        self.params['continuous'] = self.continuous.isChecked()
        self.params['verbose'] = self.verbose.isChecked()

        # Get ROI arguments
        self.params['scaler_rois'] = []
        for key in ['im', 'i0', 'it']:
            if getattr(self.roi_selector, f'scaler_{key}_button').state == 1:
                self.params['scaler_rois'].append(key)

        self.params['ignore_det_rois'] = []
        for name in ['merlin', 'dexela', 'eiger']:
            if getattr(self.roi_selector, f'{name}_button').state == 3:
                self.params['ignore_det_rois'].append(name)
        
        self.params['specific_elements'] = self.roi_selector.periodic_table.target.selection_order
        self.params['boring_elements'] = []
        self.params['excluded_elements'] = []
        for el in self.roi_selector.periodic_table.target.element_buttons:
            if not self.roi_selector.enabled[el]:
                continue
            if el.state == 2:
                self.params['boring_elements'].append(el.name)
            elif el.state == 3:
                self.params['excluded_elements'].append(el.name)

        self.params['min_roi_num'] = int(self.roi_selector.roi_controls.min_roi_num.text())
        self.params['max_roi_num'] = int(self.roi_selector.roi_controls.max_roi_num.text())

        self.worker = ReportWorker(self.params)
        self.worker.finished.connect(self._handle_finished_worker)
        self.worker.error_occured.connect(self._handle_error_worker)
        self.worker.start()


    def pause_scan_report(self, verbose=True):

        # Pause Scan report
        if self.worker and self.worker.isRunning():
            print('\nWaiting for report to pause...')
            self.worker.requestInterruption()
            self.worker.wait()
            if verbose:
                print('Report paused!')

        # Set controls states
        self._set_paused_state()


    def stop_scan_report(self):

        if not self._is_paused:
            # Sends signals to do what is needed
            self.pause_scan_report(verbose=False)

        # Stop and finalize report
        if self.worker:
            self.output = self.worker.output
            self.worker.terminate()
            self.worker = None
            print('Report finalized!')
        
        # Reset both
        self._set_idle_state()

        if self.output is not None:
            md_path, pdf_path, new_pdf_path = self.output
            
            # Finalize Scan. Code from second interrupt
            os.rename(pdf_path, new_pdf_path)
            os.remove(md_path)
            self.output = None


    def _handle_finished_worker(self):
        
        # The worker finished on its own or errored-out
        if self._is_running:
            if (self.worker is not None
                and self.worker.output is None):
                print('Report generation completed.')
                self._set_idle_state()
            else:
                # Paused/error messages are handled elswhere
                self._set_paused_state()

    
    def _handle_error_worker(self, err_str):

        print(f'Report Generation Error: {err_str}')
        # Technically should be the second pause and is redundant
        self._set_paused_state()


    def _set_running_state(self):
        if self.verbose.isChecked():
            print('Setting running state.')
        
        if self._is_running:
            if self._is_paused:
                self._is_paused = False
                self.run_button.setEnabled(False)
                self.pause_button.setEnabled(True)
                self.enable_roi_inputs(False)
                self.run_button.setText('Run')
        
        else:
            self._is_running = True
            self.run_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.enable_scan_inputs(False)
            self.enable_roi_inputs(False)


    def _set_paused_state(self, error=''):
        if self.verbose.isChecked():
            print('Setting paused state.')

        if error != '':
            print(error)
        
        # Set controls states
        self._is_paused = True
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.enable_roi_inputs(True)
        self.run_button.setText('Resume')


    def _set_idle_state(self):
        if self.verbose.isChecked():
            print('Setting idle state.')

        # Reset both
        self._is_running = False
        self._is_paused = False
        
        # Reset run button text
        self.run_button.setText('Run')

        # Reenable all buttons and inputs
        self.enable_scan_inputs(True)
        self.enable_roi_inputs(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.run_button.setEnabled(True)

        # Reset all values
        # self.clear()
        # self.roi_selector.reset_defaults()


class ReportWorker(QThread):
    # Signals to update the UI

    error_occured = pyqtSignal(str)
    current_state = 'idle'
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.output = None

    def run(self):
        try:
            # print(self.params)
            self.output = generate_scan_report(**self.params)

        # This should never occur. All exception are handled internally...
        # Useful incase this assumption is wrong
        except Exception as e:
            self.error_occured.emit(str(e))
            