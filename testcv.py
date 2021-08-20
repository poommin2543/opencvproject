
'''import cv2
import numpy as np
img = cv2.imread("./image/lines.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,75,150)

lines = cv2.HoughLinesP(edges,1,np.pi/180,20,maxLineGap=50 ,minLineLength=0)
print(lines)
for line in lines:
    x1 ,y1 ,x2 ,y2 = line[0]
    cv2.line(img,(x1 ,y1),(x2 ,y2),(0,255,0),3)

cv2.imshow("edges", edges)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
import numpy as np
import cv2
video = cv2.VideoCapture('./image/IMG_4198.MOV')


while True:
    ret, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([18, 94, 140])
    up_yellow = np.array([48, 255, 255])
    
    mask = cv2.inRange(hsv, low_yellow, up_yellow)



    if not ret:
        video = cv2.VideoCapture('./image/video1.mp4')
        continue

   
    cv2.imshow('frame', hsv)
    cv2.imshow('test',mask)

    k = cv2.waitKey(25) 
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()