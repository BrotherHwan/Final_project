import multiprocessing
import threading
from functools import partial
import os
import sys
import psutil
import time



from moviepy.editor import VideoFileClip

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import signal

import dance
import sound
import ui

import paho.mqtt.client as mqtt



class Main():    
    def __init__(self):
        self.message_que = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        self.Main_process()
        self.subjectlist = ['댄스', '야구', '축구', '농구', '골프']
        self.subject_index = 0
        self.windows = None
        self.pd = None
        
        
    def function_one(self, audio_clip):
        
        time.sleep(1)
        print("음악부분 실행완료")
        audio_clip.preview()
     
    def Main_process(self):        
        self.ui_p = multiprocessing.Process(target=self.UI_Load, args=(self.message_que, self.lock, ))
        dance_p = multiprocessing.Process(target=self.Just_Dance, args=(self.message_que, self.lock, ))
        self.sound_p = multiprocessing.Process(target=self.Sound_Control, args=(self.message_que, self.lock, ))
        
        self.ui_p.start()
        dance_p.start()
        self.sound_p.start()
        self.subject_index = 0
        global ui_pid

        
        ui_pid = self.ui_p.pid
        
        
        while 1:  #ui에서 메세지큐를 통해서 스케쥴링 조정
            if self.message_que.empty() == True:
                
                time.sleep(0.1)
            else:
                self.lock.acquire()
                key, value = self.message_que.get()
                self.lock.release()
                if key == "ui" and value == "input":
                    self.message_que.put(("sound", "input"))
                elif key == 'sound_input':
                    if value == "댄스":
                        self.subject_index = 0
                        self.message_que.put(("ui_update", self.subject_index))
                        
                    elif value == "야구":
                        self.subject_index = 1
                        self.message_que.put(("ui_update", self.subject_index))
                        
                    elif value == "농구":
                        self.subject_index = 3
                        self.message_que.put(("ui_update", self.subject_index))
                        
                    elif value == "골프":
                        self.subject_index = 4
                        self.message_que.put(("ui_update", self.subject_index))
                       
                    elif value == "축구":
                        self.subject_index = 2
                        self.message_que.put(("ui_update", self.subject_index))
                       
                    elif value == "start":
                        match self.subject_index:
                            case 0:
                                
                                video_clip = VideoFileClip("./m.mp4")
                                audio_clip = video_clip.audio
                                audio_clip = video_clip.audio
                                self.message_que.put(("dance_p","start"))
                                tp = threading.Thread(target=self.function_one, args=(audio_clip, ))
                                tp.start()
                                tp.join()    
                                
                                continue
                                
                            case 1:
                                continue
                                
                            case 2:
                                continue
                                
                            case 3:
                                continue
                                
                            case 4:
                                continue
                    
                else:
                    self.message_que.put((key, value))
                    time.sleep(0.3)


        self.ui_p.join()
        

        # self.sound_p.join()        

    def UI_Load(self, queue, lock):
     
        queue.put(("ui", "input"))
        app = QApplication(sys.argv)
        self.mainWindow = ui.Main_UI(queue)

        self.windows = self.mainWindow
        
        q_thd = threading.Thread(target=self.que_thread, args=(queue, lock, ))
        q_thd.start()

        self.mainWindow.show()
        app.exec_()
        
        
    def que_thread(self, queue, lock):
        while True:
            
            if queue.empty():
                time.sleep(0.1)
            else:
                lock.acquire()
                key, value = queue.get()
                lock.release()
                if key == 'ui_update':
                    self.mainWindow.sub_move(int(value))
                    queue.put(("sound", "input"))
                    time.sleep(0.2)
                elif key == "dance_score":
                    self.mainWindow.score_write(str(value))
                    time.sleep(1)
                    self.mainWindow.ui_reload()
                    queue.put(("sound", "input"))
                    time.sleep(3)
                else:
                    queue.put((key, value))
                    time.sleep(1)
        
    def Just_Dance(self, queue, lock):
        jd = dance.Just_Dance(queue)
        # 메인ui에서 큐에넣어주면 가져와서 해당 이름을 filename에 설정. queue.get()
        file_name = "m"
        video_file = f"./{file_name}.mp4"

        USE_WEBCAM = False
        cam_id = 0
        
        source = cam_id if USE_WEBCAM else video_file
        while 1:
            lock.acquire()
            key, value = queue.get()
            lock.release()
            if key == "dance_p" and value == "start":
                jd.run_pose_estimation(source=source, flip=False, use_popup=True, msg_queue=queue, skip_first_frames=500)
            else:
                queue.put((key,value))
                time.sleep(1)
    
        
    def Sound_Control(self, queue, lock):      
        Sound = sound.sound_reco()
        
        while 1:   
            if queue.empty()==True:
                time.sleep(0.1) 
            else :
                lock.acquire()
                key, value = queue.get()
                lock.release()
                
                if key == "sound" and value == "input":
                    
                    while 1:
                        print("음성명령 인식부분 들어옴.")
                        command = Sound.tts()
                        
                        if command == "댄스":
                            print("댄스 큐에 입력")
                            queue.put(("sound_input", "댄스"))
                            break
                        elif command == "골프":
                            print("골프 큐에 입력")
                            queue.put(("sound_input", "골프"))
                            break
                        elif command == "축구":
                            print("축구 큐에 입력")
                            queue.put(("sound_input", "축구"))
                            break
                        elif command == "농구":
                            print("농구 큐에 입력")
                            queue.put(("sound_input", "농구"))
                            break
                        elif command == "야구":
                            print("야구 큐에 입력 루프 탈출.")
                            queue.put(("sound_input", "야구"))
                            break
                        elif command == "start":
                            print("start 큐에 입력 루프 탈출.")
                            queue.put(("sound_input", "start"))
                            break
                else: 
                    queue.put((key, value))
                    print(f"key:{key}, value:{value} 큐에 다시 넣음..")
                    time.sleep(1)
                    

if __name__ == "__main__":
    
    ui_pid = None
    sound_pid = None
    
    try:
        Main()
    except KeyboardInterrupt:
        print(f"키보드 {ui_pid}")
        target = psutil.Process(ui_pid)
        target.kill()

