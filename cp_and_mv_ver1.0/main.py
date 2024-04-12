import multiprocessing
from functools import partial
import os
import sys
import psutil
import time
import subprocess

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import ui
import test

class Main():
    def __init__(self):
        self.message_que = multiprocessing.Queue()
        self.Main_process()
        
    def Main_process(self):
        ui_p = multiprocessing.Process(target = self.UI_Load, args = (self.message_que, ))
        dance_p = multiprocessing.Process(target = self.Just_Dance, args = (self.message_que, ))
        
        ui_p.start()
        dance_p.start()
        
        ui_p.join()
        dance_p.join()
        
    def UI_Load(self, queue):
        # ui.UI_Start().flag_checker(queue)
        # print(queue.get())
        app = QApplication(sys.argv)
        self.mainWindow = ui.Main_UI(queue)
        self.mainWindow.show()
        sys.exit(app.exec_())

        
    def Just_Dance(self, queue):
        flag = queue.get()
        print(flag["flag"])
        jd = test.Main()
        if flag["flag"] == "춤":
            jd.printer()
            self.mainWindow.hide()
    

if __name__ == "__main__":
    Main()
