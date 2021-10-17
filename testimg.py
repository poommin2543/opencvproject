import matplotlib.pylab as plt
import cv2
import numpy as np
import imutils
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

# Carregar modelos
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']
# perspctive + line
image = cv2.imread('./image/IMG_4743.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
copy_image = cv2.imread('./image/IMG_4743.png')
copy_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def image_detecion(img):

    width = 1920
    height = 1080

    # Resize to respect the input_shape
    inp = cv2.resize(img, (width, height))
    # inp = frame
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
        cv2.putText(img_boxes, label, (xmin, ymax - 10), font, 1, (255, 0, 0), 3, cv2.LINE_AA)
        # cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1, (255,0,0), 3, cv2.LINE_AA)

        print(ymin, xmin, ymax, xmax)
        print(xmax,xmin ,end=" = ")
        print(xmax-xmin)

        w = 1.67
        f = 0
        p = (xmax-xmin)
        d = 5

        f = (d*p)/w
        print(f)
        print(type(f))
    # Display the resulting frame
    plt.imshow(img_boxes)
    plt.show()


print(image.shape)
height = image.shape[0]
width = image.shape[1]

# ptD = [770,488]
# ptC = [1145,486]
# ptA = [25,859]
# ptB = [1756,966]
#
# wi,hi = 500,500
#
# pts1 = np.float32([ptD,ptC,ptA,ptB])
# pts2 = np.float32([[0,0],[wi,0],[0,hi],[wi,hi]])
#
# matrix = cv2.getPerspectiveTransform(pts1,pts2)
# result = cv2.warpPerspective(image,matrix,(wi,hi))


region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 3),
    (width, height)
]

IMAGE_H = 1080
IMAGE_W = 1920

points = np.array([[0, 600], [1920, 450], [1920, 1080], [0, 1080]])
print(region_of_interest_vertices)

def nothing(x):
  pass

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = (255, 0, 255)
    print(match_mask_color)
    #cv2.fillPoly(mask, vertices, match_mask_color)
    cv2.fillPoly(mask, [points], match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


cropped_image = region_of_interest(image,
                                   np.array([region_of_interest_vertices], np.int32), )
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# print(edges)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
# print(lines)

ptD = [800, 640]
ptC = [1310, 640]
ptA = [50, 1050]
ptB = [2300, 1050]

sorted_pts = np.float32([ptC, ptD, ptA, ptB])
mask = np.zeros(image.shape, dtype=np.uint8)
roi_corners = np.int32(sorted_pts)

cv2.fillConvexPoly(mask, roi_corners, (255, 255, 255))
mask = cv2.bitwise_not(mask)

    # cv2.imshow('Fused Image', mask)
    # masked_image = cv2.bitwise_and(img, mask)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 15)
    # print(line)




# ptD = [770,488]
# ptC = [1145,486]
# ptA = [25,859]
# ptB = [1756,966]
ptD = [750,640]
ptC = [1310,640]
ptA = [0,1060]
ptB = [2300,1060]

wi,hi = 1500,1500

pts1 = np.float32([ptD,ptC,ptA,ptB])
pts2 = np.float32([[0,0],[wi,0],[0,hi],[wi,hi]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
result = cv2.warpPerspective(image,matrix,(wi,hi))
# plt.imshow(result)

# a = cv2.getTrackbarPos('min','image')
# b = cv2.getTrackbarPos('max','image')
#
#
# cv2.createTrackbar('min','image',0,255,nothing)
# cv2.createTrackbar('max','image',0,255,nothing)
# thresh = cv2.threshold(result,a,b,cv2.THRESH_BINARY_INV)
# print(result)
# langht = cv2.line(result,(1250,1400),(300,1400),(255,0,0),50)
# print(langht)

font = cv2.FONT_HERSHEY_SIMPLEX
st = hi
m1 = 325#1200/3.5
i = 1

cv2.line(result,(550,st),(400,st),(255,0,0),50)
cv2.line(result,(1100,st),(950,st),(255,0,0),50)


cv2.line(result,(550,st-m1*i),(400,st-m1*i),(255,0,0),50)
cv2.line(result,(1100,st-m1*i),(950,st-m1*i),(255,0,0),50)
cv2.putText(result, '1 M', (600,st-m1*i), font, 5,
                  (255,0,0), 20, cv2.LINE_AA, False)
i+=1
cv2.line(result,(550,st-m1*i),(400,st-m1*i),(255,0,0),50)
cv2.line(result,(1100,st-m1*i),(950,st-m1*i),(255,0,0),50)
cv2.putText(result, '2 M', (600,st-m1*i), font, 5,
                  (255,0,0), 20, cv2.LINE_AA, False)
i+=1
cv2.line(result,(550,st-m1*i),(400,st-m1*i),(255,0,0),50)
cv2.line(result,(1100,st-m1*i),(950,st-m1*i),(255,0,0),50)
cv2.putText(result, '3 M', (600,st-m1*i), font, 5,
                  (255,0,0), 20, cv2.LINE_AA, False)

i+=1
cv2.line(result,(550,st-m1*i),(400,st-m1*i),(255,0,0),50)
cv2.line(result,(1100,st-m1*i),(950,st-m1*i),(255,0,0),50)
cv2.putText(result, '4 M', (600,st-m1*i), font, 5,
                  (255,0,0), 20, cv2.LINE_AA, False)

i+=1
cv2.line(result,(550,st-m1*i),(400,st-m1*i),(255,0,0),50)
cv2.line(result,(1100,st-m1*i),(950,st-m1*i),(255,0,0),50)
cv2.putText(result, '5 M', (600,st-m1*i), font, 5,
                  (255,0,0), 20, cv2.LINE_AA, False)

# print(np.linalg.inv(result))

matrix = (np.linalg.inv(matrix))
result1 = cv2.warpPerspective(result,matrix,(IMAGE_W,IMAGE_H))
output = cv2.bitwise_or(copy_image, result1)
image_detecion(output)
