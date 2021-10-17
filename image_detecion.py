import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd

# Carregar modelos
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

cap = cv2.VideoCapture('./image/Untitled.MP4')

ret, frame = cap.read()

width = 1920
height = 1080

while (True):


    # Resize to respect the input_shape
    inp = cv2.resize(frame, (width, height))
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

    # Display the resulting frame
    cv2.imshow('black and white', img_boxes)
    # Capture frame-by-frame
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()