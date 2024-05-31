import cv2
import os

def player(source):
    if os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        while True:

            rat, frame = cap.read()
            
            if not rat:
                break
            
            cv2.imshow("video", frame)
            
            if cv2.waitKey(33) == 27:
                break
        
        

if __name__ == "__main__":
    source = "./feedback_video/taekwondo.mp4"
    
    player(source)