import cv2

cap = cv2.VideoCapture(2)

while cv2.waitKey(33) < 0:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    
cap.release()

cv2.destroyAllWindows()