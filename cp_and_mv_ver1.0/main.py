import multiprocessing
import threading
from functools import partial
import os
import sys
import psutil
import time
import pymysql

from moviepy.editor import VideoFileClip

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import signal

import dance
import sound
import ui


class Main():    
    def __init__(self):
        self.message_que = multiprocessing.Queue()
        self.Main_process()
        self.subjectlist = ['댄스', '야구', '축구', '농구', '골프']
        self.subject_index = 0
        self.windows = None
        self.pd = None
        
    def function_one(self, audio_clip):
        time.sleep(1.4)
        print("음악부분 실행완료")
        audio_clip.preview()
     
    def Main_process(self):        
        self.ui_p = multiprocessing.Process(target=self.UI_Load, args=(self.message_que, ))
        dance_p = multiprocessing.Process(target=self.Just_Dance, args=(self.message_que, ))
        self.sound_p = multiprocessing.Process(target=self.Sound_Control, args=(self.message_que, ))
        
        self.ui_p.start()
        #dance_p.start()
        self.sound_p.start()
        self.subject_index = 0
        global ui_pid,sound_pid
        
        ui_pid = self.ui_p.pid
        sound_pid = self.sound_p.pid
        
        print(f"ui :{self.ui_p.pid}")
        print(f"sound_p :{self.sound_p.pid}")
        while 1:  #ui에서 메세지큐를 통해서 스케쥴링 조정
            if self.message_que.empty() == True:
                print("메인프로세스 슬립 1초")
                time.sleep(0.1)
            else:
                key, value = self.message_que.get()
                if key == "ui" and value == "input":
                    self.message_que.put(("sound", "input"))
                elif key == 'sound_input':
                    if value == "댄스":
                        self.subject_index = 0
                        self.message_que.put(("ui_update", self.subject_index))
                        continue
                    if value == "야구":
                        self.subject_index = 1
                        self.message_que.put(("ui_update", self.subject_index))
                        continue
                    if value == "농구":
                        self.subject_index = 3
                        self.message_que.put(("ui_update", self.subject_index))
                        continue
                    if value == "골프":
                        self.subject_index = 4
                        self.message_que.put(("ui_update", self.subject_index))
                        continue
                    if value == "축구":
                        self.subject_index = 2
                        self.message_que.put(("ui_update", self.subject_index))
                        continue
                    if value == "start":
                        match self.subject_index:
                            case 0:
                                video_clip = VideoFileClip("./m.mp4")
                                audio_clip = video_clip.audio
                                tp = threading.Thread(target=self.function_one, args=(audio_clip, ))
                                tp.start()
                                dance_p.start()

                                break
                                
                            case 1:
                                break
                                
                            case 2:
                                break
                                
                            case 3:
                                break
                                
                            case 4:
                                break    
                    
                else:
                    self.message_que.put((key, value))
                    time.sleep(0.3)

        tp.join()    
        self.ui_p.join()
        dance_p.join()
        self.sound_p.join()        

    def UI_Load(self, queue):
        # ui.UI_Start().flag_checker(queue)
        # print(queue.get())
        queue.put(("ui", "input"))
        app = QApplication(sys.argv)
        self.mainWindow = ui.Main_UI(queue)
        self.windows = self.mainWindow
        
        q_thd = threading.Thread(target=self.que_thread, args=(queue, ))
        q_thd.start()

        self.mainWindow.show()
        
        try:
            app.exec_()
        except KeyboardInterrupt:
            app.kill()   
        
    def que_thread(self, queue):
        while True:
            
            if queue.empty():
                time.sleep(0.3)
            else:
                key, value = queue.get()
                if key == 'ui_update':
                    self.mainWindow.sub_move(value)
                    queue.put(("sound", "input"))
                    time.sleep(2)
                    
                if key == "dance_score":
                    self.mainWindow.score_write(str(value))
                    time.sleep(5)
                    self.mainWindow.ui_reload()
                    queue.put(("sound", "input"))
                    time.sleep(1)
                    
                else:
                    queue.put((key, value))
        
    def Just_Dance(self, queue):
        jd = dance.Just_Dance(queue)
        # 메인ui에서 큐에넣어주면 가져와서 해당 이름을 filename에 설정. queue.get()
        file_name = "m"
        video_file = f"./{file_name}.mp4"

        USE_WEBCAM = False
        cam_id = 0
        
        source = cam_id if USE_WEBCAM else video_file
        
        jd.run_pose_estimation(source=source, flip=False, use_popup=True, msg_queue=queue, skip_first_frames=500)
    
        
    def Sound_Control(self, queue):      
        Sound = sound.sound_reco()
        
        while 1:   
            if queue.empty()==True:
                time.sleep(0.5) 
            else :
                key, value = queue.get()
                if key=='quit' and value=='all':
                    
                    sys.exit()
                elif key == "sound" and value == "input":
                    
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
                    
    def db_insert(self, name, score):
        
        db_config = {
                'host': '10.10.52.141',
                'port':3306,
                'user': 'jsum',
                'password': '11110000',
                'database': 'dance_record'
                }
        
        try:
            conn = pymysql.connect(**db_config)
            # MySQL 서버에 연결
            
            if conn.open:
                print('MySQL 서버에 연결되었습니다.')
            else:
                print('MySQL 서버에 연결할 수 없습니다.')
                
                
                ###########  기록 삽입
                # 데이터 삽입을 위한 SQL 쿼리
            insert_query = "INSERT INTO dance_score (id, score) VALUES (%s, %s)"
            update_query = "UPDATE dance_score SET score = %s WHERE id = %s"

            # 데이터 삽입할 값
            data_to_insert = ('test1', '101')

            # 커서 생성
            cursor = conn.cursor()

            # 쿼리 실행 (인서트)
            # cursor.execute(insert_query, data_to_insert)
            new_score = 95  # 새로운 점수
            username = 'test1'  # 업데이트할 사용자 이름

            # 쿼리 실행
            cursor.execute(update_query, (new_score, username))

            # 변경사항 커밋
            conn.commit()

            print(f'{cursor.rowcount}개의 레코드가 삽입되었습니다.')

        # 커서 및 연결 닫기
            cursor.close()
            conn.close()
        except pymysql.Error as e:
            print(f'MySQL 에러 발생: {e}')

        
    
        


if __name__ == "__main__":
    
    ui_pid = None
    sound_pid = None
    
    try:
        my_pro = Main()
    except KeyboardInterrupt:
        print(f"키보드 {ui_pid}")
        target = psutil.Process(ui_pid)
        target.kill()

