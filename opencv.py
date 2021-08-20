import numpy as np
import cv2
import moviepy.editor as mp
cap = cv2.VideoCapture('./image/IMG_34731.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

def find_street_lanes(image):
    grayscale_image = grayscale(image)
    blur_image = blur(grayscale_image)
    canny_image = canny(blur_image)
    roi_image = roi(canny_image)
    hough_lines_image = hough_lines(roi_image, 0.9, np.pi/180, 100, 100, 50)
    final_image = combine_images(hough_lines_image, image)


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