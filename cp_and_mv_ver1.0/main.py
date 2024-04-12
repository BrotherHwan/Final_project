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
import dance
import sound

class Main():
    def __init__(self):
        self.message_que = multiprocessing.Queue()
        self.Main_process()
        
    def Main_process(self):
        ui_p = multiprocessing.Process(target = self.UI_Load, args = (self.message_que, ))
        dance_p = multiprocessing.Process(target = self.Just_Dance, args = (self.message_que, ))
        sound_p = multiprocessing.Process(target = self.Sound_Control, args = (self.message_queue, ))
        
        ui_p.start()
        dance_p.start()
        sound_p.start()
        
        ui_p.join()
        dance_p.join()
        sound_p.join()
        
        
    def UI_Load(self, queue):
        # ui.UI_Start().flag_checker(queue)
        # print(queue.get())
        app = QApplication(sys.argv)
        self.mainWindow = ui.Main_UI(queue)
        self.mainWindow.show()
        
        flag = queue.get()
        queue.put({"flag":flag["flag"]})
        
        if flag["falg"] == "춤":
            self.mainWindow.hide()
            
        sys.exit(app.exec_())

        
    def Just_Dance(self, queue):
        flag = queue.get()
        print(flag["flag"])
        jd = dance.Just_Dance()
        if flag["flag"] == "춤":
            USE_WEBCAM = False
            cam_id = 0
            video_file = "./m.mp4"
            source = cam_id if USE_WEBCAM else video_file
            additional_options = {"skip_first_frames": 500} if not USE_WEBCAM else {}
            jd.run_pose_estimation(source=source, flip=False, use_popup=True, **additional_options)

            
    def Sound_Control(self, queue):
        self.sound = sound.sound_reco(queue)
        self.sound.tts()
            

if __name__ == "__main__":
    Main()
