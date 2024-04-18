import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import time
import multiprocessing
from functools import partial

import signal


main_window = uic.loadUiType('./Just_dance.ui')[0]

class Main_UI(QWidget, main_window):
    option = ""
    
    def __init__(self, msg_queue):
        super().__init__()
        self.setupUi(self)
        
        self.pos = [260, 310, 360, 410, 460]
        self.pos_idx = 0
        
        self.txt_title.setHidden(True)
        self.txt_menu.setHidden(True)
        self.txt_info.setHidden(True)
        # self.edit_option.setHidden(True)
        self.label.setHidden(True)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.ui_change)
        self.timer.setSingleShot(True)
        self.timer.start(3000)
        
        # self.edit_option.returnPressed.connect(partial(self.option_select, msg_queue))
        
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
        # self.edit_option.setHidden(False)
        self.label.setHidden(False)
        
    def ui_reload(self):
        self.txt_title.setHidden(False)
        self.txt_welcome.setHidden(True)
        self.txt_menu.setHidden(False)
        self.txt_info.setHidden(False)
        self.label.setHidden(False)

    def option_select(self, msg_queue):
        self.option = self.edit_option.text()
        # print(self.option)
        msg_queue.put({"flag":self.option})
        # self.edit_option.setText("")
        
        if self.option == "춤":
            # subprocess.run(["python", "JUST_DANCE_FINAL.py"])
            pass
    
    def sub_move(self, idx):
        self.label.move(90, self.pos[idx])

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up:
            self.pos_idx = self.pos_idx - 1
            if self.pos_idx < 0:
                self.pos_idx = 2

        elif e.key() == Qt.Key_Down:
            self.pos_idx = self.pos_idx + 1
            if self.pos_idx > len(self.pos)-1:
                self.pos_idx = 0

        self.label.move(90, self.pos[self.pos_idx])

    def score_write(self, score):
        self.txt_title.setHidden(True)
        self.txt_menu.setHidden(True)
        self.txt_info.setHidden(True)
        self.label.setHidden(True)
        self.txt_welcome.setHidden(False)
        self.txt_welcome.setText(str(score))
        
# class UI_Start():
#     def __init__(self):
#         app = QApplication(sys.argv)
#         self.mainWindow = Main_UI()
#         self.mainWindow.show()
#         sys.exit(app.exec_())
    
#     def flag_checker(self, queue):
#         print(self.mainWindow.option)
#     #     queue.put({"flag" : self.mainWindow.option})
        
#     def __del__(self):
#         print("ui 종료")
        
        
if __name__ == '__main__':
    def sig_hadler(signal, frame):
        sys.exit()
    
    
    app = QApplication(sys.argv)
    mainWindow = Main_UI(None)
    mainWindow.show()
    app.exec_()
    
    signal.signal(signal.SIGINT, sig_hadler)

    
    # UI_Start()
    