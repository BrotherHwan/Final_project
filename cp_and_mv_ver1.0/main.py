import multiprocessing
from functools import partial
import os
import sys
# import psutil
import time
import subprocess

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
        ui.UI_Start()
        
    def Just_Dance(self, queue):
        flag = queue.get()
        test.Main().printer(flag)
    

if __name__ == "__main__":
    Main()
