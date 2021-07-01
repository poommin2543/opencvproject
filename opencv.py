import numpy as np
import cv2
import moviepy.editor as mp
cap = cv2.VideoCapture('./image/IMG_34731.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    frameb = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    fgmask = fgbg.apply(frameb)

    cv2.imshow('fgmask', fgmask)
    cv2.imshow('frameb', frameb)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()