import matplotlib.pylab as plt
import cv2
import numpy as np
import imutils
# perspctive + line
image = cv2.imread('IMG_4244.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

points = np.array([[0, 450], [1920, 450], [1920, 1080], [0, 1080]])
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
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 15)
    # print(line)


# ptD = [770,488]
# ptC = [1145,486]
# ptA = [25,859]
# ptB = [1756,966]
ptD = [700,500]
ptC = [1200,486]
ptA = [0,800]
ptB = [2000,966]

wi,hi = 1500,1500

pts1 = np.float32([ptD,ptC,ptA,ptB])
pts2 = np.float32([[0,0],[wi,0],[0,hi],[wi,hi]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
result = cv2.warpPerspective(image,matrix,(wi,hi))


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
st = 1500
cv2.line(result,(350,st),(150,st),(255,0,0),50)
cv2.line(result,(1400,st),(1200,st),(255,0,0),50)


cv2.line(result,(350,st-340),(150,st-340),(255,0,0),50)
cv2.line(result,(1400,st-340),(1200,st-340),(255,0,0),50)
cv2.putText(result, '1 M', (600,st-340), font, 5,
                  (255,0,0), 20, cv2.LINE_AA, False)

cv2.line(result,(350,st-340*2),(150,st-340*2),(255,0,0),50)
cv2.line(result,(1400,st-340*2),(1200,st-340*2),(255,0,0),50)
cv2.putText(result, '2 M', (600,st-340*2), font, 5,
                  (255,0,0), 20, cv2.LINE_AA, False)

cv2.line(result,(350,st-340*3),(150,st-340*3),(255,0,0),50)
cv2.line(result,(1400,st-340*3),(1200,st-340*3),(255,0,0),50)
cv2.putText(result, '3 M', (600,st-340*3), font, 5,
                  (255,0,0), 20, cv2.LINE_AA, False)
# print(np.linalg.inv(result))
plt.imshow(result)
# plt.imshow(image)
plt.show()