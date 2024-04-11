import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *


import os
import time
import subprocess


main_window = uic.loadUiType('./Just_dance.ui')[0]

class Main_UI(QWidget, main_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.txt_title.setHidden(True)
        self.txt_menu.setHidden(True)
        self.txt_info.setHidden(True)
        self.edit_option.setHidden(True)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.ui_change)
        self.timer.setSingleShot(True)
        self.timer.start(3000)
        
        self.edit_option.returnPressed.connect(self.option_select)
        
    def ui_change(self):
        self.txt_welcome.setHidden(True)
        self.txt_title.setHidden(False)
        
        temp_tim = QTimer(self)
        temp_tim.timeout.connect(self.ui_change2)
        temp_tim.setSingleShot(True)
        temp_tim.start(500)
        
    def ui_change2(self):
        self.txt_menu.setHidden(False)
        self.txt_info.setHidden(False)
        self.edit_option.setHidden(False)
        
    def option_select(self):
        option = self.edit_option.text()
        print(option)
        
        self.edit_option.setText("")
        
        if option == "춤":
            subprocess.run(["python", "JUST_DANCE_FINAL.py"])
        
        
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Main_UI()
    mainWindow.show()
    sys.exit(app.exec_())
