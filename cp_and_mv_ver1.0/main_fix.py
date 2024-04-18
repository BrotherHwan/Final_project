import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *


import os
import time
import multiprocessing
from functools import partial

import sound
import dance

main_window = uic.loadUiType('./Just_dance.ui')[0]

class Main_UI(QWidget, main_window):
    option = ""
    
    def __init__(self, msg_queue):
        super().__init__()
        self.setupUi(self)
        
        self.pos = [260, 310, 360]
        self.pos_idx = 0
        
        self.txt_title.setHidden(True)
        self.txt_menu.setHidden(True)
        self.txt_info.setHidden(True)
        self.edit_option.setHidden(True)
        self.label.setHidden(True)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.ui_change)
        self.timer.setSingleShot(True)
        self.timer.start(3000)
        
        self.edit_option.returnPressed.connect(partial(self.option_select, msg_queue))
        
        
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
        self.label.setHidden(False)
        
        
    def option_select(self, msg_queue):
        self.option = self.edit_option.text()
        # print(self.option)
        msg_queue.put({"flag":self.option})
        self.edit_option.setText("")
        
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
        

class Main_Process():
    def __init__(self):
        app = QApplication(sys.argv)
        mainWindow = Main_UI(None)
        mainWindow.show()
        
        self.message_que = multiprocessing.Queue()
        self.subjectlist=['춤','야구','축구','농구','골프']
        self.subject_index=-1
        
        dance_p = multiprocessing.Process(target = self.Just_Dance, args = (self.message_que, ))
        sound_p = multiprocessing.Process(target = self.Sound_Control,args = (self.message_que, ))

        sound_p.start()
        
        sound_p.join()
        
        sys.exit(app.exec_())
        
    
    def Just_Dance(self, queue):
        
        jd = dance.Just_Dance(queue)
        #메인ui에서 큐에넣어주면 가져와서 해당 이름을 filename에 설정. queue.get()
        file_name="m"
        video_file = f"./{file_name}.mp4"
       
        USE_WEBCAM = False
        cam_id = 0
        
        source = cam_id if USE_WEBCAM else video_file
        
        jd.run_pose_estimation(source=source, flip=False, use_popup=True, msg_queue = queue, skip_first_frames=500)
        
        
    def Sound_Control(self,queue):
        Sound = sound.sound_reco()
        
        while 1:    
            

            key, value = queue.get()
            print(f"sound_control에서 key:{key}, value:{value} 큐에서 받음.")
            if key == "sound" and value =="input":
                while 1:
                    print("음성명령 인식부분 들어옴.")
                    command = Sound.tts()
                    if command == "춤":
                        print("춤 큐에 입력")
                        queue.put(("sound_input","춤"))
                        break
                    elif command == "골프":
                        print("골프 큐에 입력")
                        queue.put(("sound_input","골프"))
                        break
                    elif command == "농구":
                        print("농구 큐에 입력")
                        queue.put(("sound_input","농구"))
                        break
                    elif command == "야구":
                        print("야구 큐에 입력 루프 탈출.")
                        queue.put(("sound_input","야구"))
                        break
                    elif command =="start" :
                        print("start 큐에 입력 루프 탈출.")
                        queue.put(("sound_input","start"))
            else:
                
                queue.put((key,value))
                print(f"key:{key}, value:{value} 큐에 다시 넣음..")
                time.sleep(0.3)   
                
        
if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # mainWindow = Main_UI(None)
    # mainWindow.show()
    # sys.exit(app.exec_())
    
    # UI_Start()
    
    Main_Process()
    
    
    
    