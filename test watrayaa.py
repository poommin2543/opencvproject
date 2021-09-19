import numpy as np
import imutils
import cv2
import sys
import tkinter as tk
from tkinter import filedialog
import time


def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.Canny(thresh, 10, 30)

    # find the contours in the edged image and keep the largest one;
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


def imshow(pic):
    cv2.imshow("Output", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# initialize known parameter
KNOWN_DISTANCE = 0.75
KNOWN_WIDTH = 0.21
WriteFile = False  #Write output to video file

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
if not file_path:
    sys.exit("Please select folder of image...")

# load test video
vs = cv2.VideoCapture(file_path)
time.sleep(2.0)

ret, frs = vs.read()
marker = find_marker(frs)
P = marker[1][0]  #width in px
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# Write video file

if WriteFile:
    h, w = frs.shape[:2]
    fource = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('outpy.mp4', fource, 20, (w, h))

while True:
    ret, frame = vs.read()
    if ret == False:
        break

    marker = find_marker(frame)
    meters = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    cv2.putText(frame, "%.2f m" % (meters), (frame.shape[1] - 300, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

    cv2.imshow("image", frame)
    if WriteFile:
        out.write(frame)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
if WriteFile:
    out.release()
cv2.destroyAllWindows()