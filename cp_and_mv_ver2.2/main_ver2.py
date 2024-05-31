import multiprocessing
import threading
from functools import partial
import os
import sys
import psutil
import time

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import dance
import ui
import hand_control


class Main():
    def __init__(self):
        self.msg_que = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        self.subject_list = ["dance", "taekwondo", "feedback"]
        self.subject_idx = 0
        
    
    def Main_Process(self):
        self.msg_que.put(("start", None))
        time.sleep(1)
        
        global ui_pid
        
        ui_proc = multiprocessing.Process(target=self.UI_Process, args=(self.msg_que, self.lock, ))
        hand_proc = multiprocessing.Process(target=self.Hand_Process, args=(self.msg_que, self.lock, ))
        dance_proc = multiprocessing.Process(target=self.Dance_Process, args=(self.msg_que, self.lock, ))
        
        ui_proc.start()
        hand_proc.start()
        dance_proc.start()
        
        ui_pid = ui_proc.pid
        
        # if something:  # 특정 커멘드 일때 dance proc 실행
        #     
        
        ui_proc.join()
        hand_proc.join()
        dance_proc.join()
        
    
    def UI_Process(self, queue, lock):
        app = QApplication(sys.argv)
        self.mainWindow = ui.Main_UI(queue, lock)
        temp_idx = 0
        q_thd = threading.Thread(target=self.que_thread, args=(queue, lock, temp_idx, ))
        q_thd.start()
        
        self.mainWindow.show()
        
        app.exec_()
        
    def que_thread(self, queue, lock, temp_idx):
        current = ""
        
        l_flag = False
        r_flag = False
        
        while True:
            lock.acquire()
            target, value = queue.get()
            lock.release()
            print(f"in ui : {target}, {value}, {temp_idx}")
            if target == 'gesture':
                if value == "left":                           
                    temp_idx += 1
                    
                    if temp_idx > 2:
                        temp_idx = 0
                            
                elif value == "right":
                    temp_idx -= 1
                    
                    if temp_idx < 0:
                        temp_idx = 2 
                            
                elif value == "shortcut1":
                    self.mainWindow.sub_move(0)
                    temp_idx = 0
                    
                elif value == "shortcut2":
                    self.mainWindow.sub_move(1)
                    temp_idx = 1
                    
                elif value == "select":
                    time.sleep(0.5)
                    print(self.subject_list[temp_idx])
                    if self.subject_list[temp_idx] != "feedback":
                        queue.put((self.subject_list[temp_idx], "start"))
                        time.sleep(1)
                    else:
                        self.mainWindow.feedback_on()
                elif value == "exit":
                    queue.put(("proc", "end"))
                    break               
                self.mainWindow.sub_move(int(temp_idx))
                
                time.sleep(1)
                
            elif target == "dance_score":
                self.mainWindow.score_write(str(value))
                time.sleep(1)
                self.mainWindow.ui_reload()
                
                queue.put(("control", "on")) 
                time.sleep(1)   
                 
            elif target is not None and value is not None:
                queue.put((target, value))
                time.sleep(1)
            
            elif target is None:
                queue.put(("start", None))
                time.sleep(1)                
    
    def Hand_Process(self, queue, lock):
        
        # print(f"in hand proc queue : {queue.empty()}")
        
        # hand_controller = hand_control.Hand_Control(queue, lock)
        
        hand_controller = hand_control.Hand_Control(queue, lock)
        
        while True:
            lock.acquire()
            target, cmd = queue.get()
            print(f"in hand proc : {target}, {cmd}")
            lock.release()
            
            if target == "start":
                hand_controller.run_predic(None, queue)
            elif target == "control":
                cam = hand_controller.make_cam()
                hand_controller.run_predic(cam, queue)
            elif target == "proc" and cmd == "end":
                queue.put((target, cmd))
                time.sleep(1)
                break
            else:
                queue.put((target, cmd))
                time.sleep(1)
        
    
    def Dance_Process(self, queue, lock):
        jd = dance.Just_Dance(queue)
        
        while True:
            lock.acquire()
            target, cmd = queue.get()
            print(f"in dance proc : {target}, {cmd}")
            lock.release()
            
            if target in self.subject_list:
                if cmd == "start":
                    vid_source = f"./videos/{target}.mp4"
                
                    jd.run_pose_estimation(source=vid_source, flip=False, use_popup=True,
                                        msg_queue=queue, skip_first_frames=500, index=0)
            elif target == "proc":
                queue.put((target, cmd))
                break
            else:
                queue.put((target, cmd))
                time.sleep(1)
    
    
if __name__ == "__main__":
    ui_pid = None
    
    try:
        main_work = Main()
        main_work.Main_Process()
    except KeyboardInterrupt:
        target = psutil.Process(ui_pid)  # ui에 대한 프로세스 아이디를 지정 
        target.kill()  # ui 프로세서 죽이기