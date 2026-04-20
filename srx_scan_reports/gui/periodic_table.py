


from PyQt5.QtCore import Qt, pyqtSignal, QRect
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QPushButton,
    QSizePolicy,
    QSpacerItem
    )
from PyQt5.QtGui import (
    QPainter,
    QFont,
    QColor,
    QPen,
    QPalette
)


class ToggleButton(QPushButton):

    stateChanged = pyqtSignal(str, int)

    def __init__(self, name, states=[0, 1, 2, 3]):
        super().__init__(name)

        self.name = name
        self.states = states
        self.state = 0
        self.display_index = None
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(True) 

        self._colors = {
            0 : self.palette().color(self.backgroundRole()), # Maybe
            1 : QColor(144, 238, 144), # Include
            2 : QColor(225, 215, 160), # Ignore
            3 : QColor(255, 128, 128) # Exclude
        }


    def mousePressEvent(self, event):
        old_state = self.state
        # Left Click Logic
        if event.button() == Qt.LeftButton:
            cur_idx = self.states.index(self.state)
            self.state = self.states[(cur_idx + 1) % len(self.states)]
            
        # Right Click Logic
        elif event.button() == Qt.RightButton:
            cur_idx = self.states.index(self.state)
            self.state = self.states[(cur_idx - 1) % len(self.states)]

        if old_state != self.state:
            self.stateChanged.emit(self.name, self.state)
        
        self.updateBackgroundColor()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Calculate font size as a percentage of button height
        new_font_size = max(8, int(self.height() * 0.3)) 
        font = self.font()
        font.setPixelSize(new_font_size)
        self.setFont(font)


    def updateBackgroundColor(self):
        palette = self.palette()
        
        palette.setColor(QPalette.Button, self._colors[self.state])
        
        self.setPalette(palette)


    def paintEvent(self, event):
        # 1. Draw the button normally (the name and background)
        super().paintEvent(event)

        # 2. If it's in the "Include" state and has an index, draw the badge
        if self.state == 1 and self.display_index is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Define badge size based on button height
            badge_size = int(self.height() * 0.25)
            badge_rect = QRect(self.width() - badge_size - 2, 2, badge_size, badge_size)

            # Draw badge background (small dark circle or square)
            painter.setBrush(QColor(0, 0, 0, 100)) # Semi-transparent black
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(badge_rect, 3, 3)

            # Draw the index number
            painter.setPen(QPen(Qt.white))
            font = painter.font()
            font.setPixelSize(int(badge_size * 0.8))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(badge_rect, Qt.AlignCenter, str(self.display_index))
            painter.end()


class AspectRatioWidget(QWidget):
    def __init__(self, widget, ratio):
        super().__init__()
        self.target = widget
        self.target.setParent(self)
        self.ratio = ratio
        
        # This tells the layout to honor height-for-width calculations
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def heightForWidth(self, width):
        return int(width / self.ratio)

    def hasHeightForWidth(self):
        return True

    def resizeEvent(self, event):
        s = event.size()
        w, h = s.width(), s.height()

        # Calculate dimensions based on available space and ratio
        if w / h > self.ratio:
            new_w = int(h * self.ratio)
            new_h = h
        else:
            new_w = w
            new_h = int(w / self.ratio)

        # Center the table within the window
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        self.target.setGeometry(x, y, new_w, new_h)



class QPeriodicTable(QWidget):

    def __init__(self):
        super().__init__()

        self.selection_order = []

        self.init_UI()
        self.SRX_presets()

    
    def init_UI(self):

        self.grid = QGridLayout()
        self.grid.setContentsMargins(25, 25, 25, 25)
        self.grid.setSpacing(0)
        self.element_buttons = []

        # Column 1: Alkali Metals
        col_ind = 0
        start_row = 0
        for row_ind, el in enumerate(['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, start_row + row_ind, col_ind)


        # Column 2: Alkaline Earth Metals
        col_ind = 1
        start_row = 1
        for row_ind, el in enumerate(['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, start_row + row_ind, col_ind)

        # Column 13: Boron Group
        col_ind = 12
        start_row = 1
        for row_ind, el in enumerate(['B', 'Al', 'Ga', 'In', 'Tl', 'Nh']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, start_row + row_ind, col_ind)

        # Column 14: Carbon Group
        col_ind = 13
        start_row = 1
        for row_ind, el in enumerate(['C', 'Si', 'Ge', 'Sn', 'Pb', 'Fl']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, start_row + row_ind, col_ind)

        # Column 15: Pnictogens
        col_ind = 14
        start_row = 1
        for row_ind, el in enumerate(['N', 'P', 'As', 'Sb', 'Bi', 'Mc']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, start_row + row_ind, col_ind)

        # Column 16: Chalcogens
        col_ind = 15
        start_row = 1
        for row_ind, el in enumerate(['O', 'S', 'Se', 'Te', 'Po', 'Lv']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, start_row + row_ind, col_ind)

        # Column 17: Halogens
        col_ind = 16
        start_row = 1
        for row_ind, el in enumerate(['F', 'Cl', 'Br', 'I', 'At', 'Ts']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, start_row + row_ind, col_ind)

        # Column 17: Noble Gases
        col_ind = 17
        start_row = 0
        for row_ind, el in enumerate(['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, start_row + row_ind, col_ind)

        # Transition Metals Row 1
        row_ind = 3
        for col_ind, el in enumerate(['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, row_ind, 2 + col_ind)

        # Transition Metals Row 2
        row_ind = 4
        for col_ind, el in enumerate(['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, row_ind, 2 + col_ind)

        # Transition Metals Row 3
        row_ind = 5
        for col_ind, el in enumerate(['La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, row_ind, 2 + col_ind)

        # Transition Metals Row 4
        row_ind = 6
        for col_ind, el in enumerate(['Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, row_ind, 2 + col_ind)

        # # Add space
        self.grid.addWidget(QWidget(), 7, 0)

        # Lanthanides
        row_ind = 8
        for col_ind, el in enumerate(['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, row_ind, 3 + col_ind)

        # Actinides
        row_ind = 9
        for col_ind, el in enumerate(['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']):
            widget = ToggleButton(el)
            setattr(self, el, widget)
            self.element_buttons.append(widget)
            self.grid.addWidget(widget, row_ind, 3 + col_ind)

        for i in range(18):
            self.grid.setColumnStretch(i, 1)
            self.grid.setColumnMinimumWidth(i, 20) # Prevents columns from disappearing

        for i in range(10):
            self.grid.setRowStretch(i, 1)
            self.grid.setRowMinimumHeight(i, 20) # Prevents columns from disappearing

        # Connect all elements
        for el in self.element_buttons:
            el.stateChanged.connect(self.handle_element_toggle)

        self.setLayout(self.grid)

    
    def handle_element_toggle(self, name, state):
        if state == 1:
            if name not in self.selection_order:
                self.selection_order.append(name)
        else:
            if name in self.selection_order:
                self.selection_order.remove(name)
        
        self.update_order_display()

    
    def update_order_display(self):
        # Loop through all buttons and update their internal index
        for btn in self.element_buttons:
            if btn.name in self.selection_order:
                # Store the 1-based index (position in list + 1)
                btn.display_index = self.selection_order.index(btn.name) + 1
            else:
                btn.display_index = None
            
            # Force the button to redraw with the new number
            btn.update() 
    

    def SRX_presets(self):

        self._too_low_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al']
        self._unstable_elements = ['Og', 'Ts', 'Lv', 'Mc', 'Fl', 'Nh', 'Cn', 'Rg', 'Ds', 'Mt', 'Hs', 'Bh', 'Sg', 'Db', 'Rf', 'Lr', 'No', 'Md', 'Fm', 'Es']
        self._too_boring_elements = ['Ar']
        self._uncommon_elements = ['Tc', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf']
        
        self._disable_elements = (self._too_low_elements
                                  + self._unstable_elements)

        # Remove and disable
        for el in self._disable_elements:
            btn = getattr(self, el)
            btn.state = 3
            btn.updateBackgroundColor()
            btn.setEnabled(False)
        
        # Ignore
        for el in self._too_boring_elements:
            btn = getattr(self, el)
            btn.state = 2
            btn.updateBackgroundColor()

        # Remove
        for el in self._uncommon_elements:
            btn = getattr(self, el)
            btn.state = 3
            btn.updateBackgroundColor()