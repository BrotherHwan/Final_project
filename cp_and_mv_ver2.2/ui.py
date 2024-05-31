import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *


import os
import time
import multiprocessing
from functools import partial
import glob

import signal


main_window = uic.loadUiType('./Just_dance.ui')[0]

class Main_UI(QWidget, main_window):
    option = ""
    
    def __init__(self, msg_queue, lock):
        super().__init__()
        self.setupUi(self)
        
        self.pos = [430, 470, 520]
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
        
        self.msg_queue = msg_queue
        self.lock = lock
        
        # self.edit_option.returnPressed.connect(partial(self.option_select, msg_queue))
        
        # for video list
        self.video_lst = QListWidget()
        # self.video_lst.setGeometry(0, 0, 440, 1060)
        
        # for file in glob.glob("./feedback_video/*.mp4"):
        #     print(file)
        #     self.video_lst.addItem(file)

        self.video_lst.itemClicked.connect(self.item_clicked)
        
        # for exit
        self.exit_btn = QPushButton("exit")
        self.exit_btn.clicked.connect(self.feedback_off)
        
        # for vertical layout
        self.vlayout = QVBoxLayout()
        
        self.vlayout.addWidget(self.video_lst)
        self.vlayout.setStretchFactor(self.video_lst, 7)
        self.vlayout.addWidget(self.exit_btn)
        self.vlayout.setStretchFactor(self.exit_btn, 3)
        
        self.video_widget = QVideoWidget()
        self.video_widget.setGeometry(0, 0, 1080, 720)
        
        self.setup_player()
        
        layout = QHBoxLayout()
        layout.addWidget(self.video_widget)
        layout.setStretchFactor(self.video_widget, 3)
        layout.addLayout(self.vlayout)
        layout.setStretchFactor(self.vlayout, 1)

        self.media_group.setLayout(layout)
        
        self.media_group.setHidden(True)
        
    def setup_player(self):
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)   
    
    def item_clicked(self, item):
        work_dir = "/home/max/Desktop/Final_project/cp_and_mv_ver2.2"
        print(f"video : {work_dir + item.text()[1:]}")
        file2open = work_dir + item.text()[1:]
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file2open)))
        self.player.play()
    
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
        self.label.move(740, self.pos[idx])
        
    def feedback_on(self):
        for file in glob.glob("./feedback_video/*.mp4"):
            print(file)
            self.video_lst.addItem(file)
        
        self.media_group.setHidden(False)
    
    def feedback_off(self):
        self.media_group.setHidden(True)

    def keyPressEvent(self, e):        
        if e.key() == Qt.Key_Up:
            self.pos_idx = self.pos_idx - 1
            if self.pos_idx < 0:
                self.pos_idx = 2

        elif e.key() == Qt.Key_Down:
            self.pos_idx = self.pos_idx + 1
            if self.pos_idx > len(self.pos)-1:
                self.pos_idx = 0
                
        elif e.key() == Qt.Key_Escape:
            if self.msg_queue is not None:
                self.msg_queue.put(("proc", "end"))
            self.close()
            
        elif e.key() == Qt.Key_Left:
            self.media_group.setHidden(False)
            
        elif e.key() == Qt.Key_Right:
            self.media_group.setHidden(True)
                
        self.label.move(740, self.pos[self.pos_idx])

    def score_write(self, score):
        self.txt_title.setHidden(True)
        self.txt_menu.setHidden(True)
        self.txt_info.setHidden(True)
        self.label.setHidden(True)
        self.txt_welcome.setHidden(False)
        self.txt_welcome.setText(str(score) + "점")
        
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
    mainWindow = Main_UI(None, None)
    mainWindow.show()
    app.exec_()
    
    signal.signal(signal.SIGINT, sig_hadler)

    # UI_Start()
    