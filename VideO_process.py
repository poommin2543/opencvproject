import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import winsound

# Carregar modelos
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

def Beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 50  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
points = np.array([[0, 450], [1920, 450], [1920,1080], [0, 1080]])

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color=(255,0,255)
    cv2.fillPoly(mask, [points], match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def image_detecion(img):
    width = 1920
    height = 1080
    # Resize to respect the input_shape
    inp = cv2.resize(img, (width, height))
    # Convert img to RGB
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    # Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)
    boxes, scores, classes, num_detections = detector(rgb_tensor)
    pred_labels = classes.numpy().astype('int')[0]

    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    # loop throughout the faces detected and place a box around it
    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue

        score_txt = f'{100 * round(score, 0)}'
        img_boxes = cv2.rectangle(inp, (xmin, ymax), (xmax, ymin), (0, 255, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Mitsubishi Attrage 2016 1.670 m
        w = 1.67
        f = 401.1976047904192
        p = (xmax - xmin)
        d = 0

        d = (f*w)/p

        d = str(d)
        d = d[0:4]


        cv2.putText(img_boxes, str(d) + "Meter", (xmin, ymin - 10), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        if float(d) <= 3 :
            print("=" * 20)
            print(label, d + " Meter")
            label1 = label + " " + d + " Meter !!!"
            cv2.putText(img_boxes, label1, (300, 200), font, 5, (0, 0, 255), 10, cv2.LINE_AA)
            Beep()
    # Display the resulting frame
    cv2.imshow('black and white', img_boxes)



def process(image):
    copy_image = image
    cropped_image = region_of_interest(image,
                                       np.array([points], np.int32), )
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=100)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    ptD = [750, 640]
    ptC = [1310, 640]
    ptA = [0, 1060]
    ptB = [2300, 1060]

    wi, hi = 1500, 1500
    pts1 = np.float32([ptD, ptC, ptA, ptB])
    pts2 = np.float32([[0, 0], [wi, 0], [0, hi], [wi, hi]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (wi, hi))

    font = cv2.FONT_HERSHEY_SIMPLEX
    st = hi
    m1 = 325  # 1200/3.5
    i = 1

    cv2.line(result, (550, st), (400, st), (255, 0, 0), 50)
    cv2.line(result, (1100, st), (950, st), (255, 0, 0), 50)

    cv2.line(result, (550, st - m1 * i), (400, st - m1 * i), (255, 0, 0), 50)
    cv2.line(result, (1100, st - m1 * i), (950, st - m1 * i), (255, 0, 0), 50)
    cv2.putText(result, '1 M', (600, st - m1 * i), font, 5,
                (255, 0, 0), 20, cv2.LINE_AA, False)
    i += 1
    cv2.line(result, (550, st - m1 * i), (400, st - m1 * i), (255, 0, 0), 50)
    cv2.line(result, (1100, st - m1 * i), (950, st - m1 * i), (255, 0, 0), 50)
    cv2.putText(result, '2 M', (600, st - m1 * i), font, 5,
                (255, 0, 0), 20, cv2.LINE_AA, False)
    i += 1
    cv2.line(result, (550, st - m1 * i), (400, st - m1 * i), (255, 0, 0), 50)
    cv2.line(result, (1100, st - m1 * i), (950, st - m1 * i), (255, 0, 0), 50)
    cv2.putText(result, '3 M', (600, st - m1 * i), font, 5,
                (255, 0, 0), 20, cv2.LINE_AA, False)

    i += 1
    cv2.line(result, (550, st - m1 * i), (400, st - m1 * i), (255, 0, 0), 50)
    cv2.line(result, (1100, st - m1 * i), (950, st - m1 * i), (255, 0, 0), 50)
    cv2.putText(result, '4 M', (600, st - m1 * i), font, 5,
                (255, 0, 0), 20, cv2.LINE_AA, False)

    result = np.array(result)
    IMAGE_H = 1080
    IMAGE_W = 1920

    matrix = (np.linalg.inv(matrix))
    result1 = cv2.warpPerspective(result, matrix, (IMAGE_W, IMAGE_H))
    # output = cv2.bitwise_or(copy_image, result1)
    return result1


def output(img,mark):
    ptD = [750, 640]
    ptC = [1310, 640]
    ptA = [0, 1060]
    ptB = [2300, 1060]

    sorted_pts = np.float32([ptC, ptD, ptA, ptB])
    mask = np.zeros(mark.shape, dtype=np.uint8)

    roi_corners = np.int32(sorted_pts)
    # print(sorted_pts)

    cv2.fillConvexPoly(mask, roi_corners, (255, 255, 255))
    mask = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(img, mask)
    output = cv2.bitwise_or(mark, masked_image)
    image_detecion(output)


cap = cv2.VideoCapture('./image/Untitled.MP4')
cap1 = cap
success, img = cap.read()
success1, img1 = cap.read()
i = 0
while success:
    try:
        output(img1, process(img))
    except:
        if i > 700:
            break
        # print("error")
        i+=1
        continue
    success, img = cap.read()
    success1, img1 = cap.read()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
