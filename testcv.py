import matplotlib.pylab as plt
import cv2
import numpy as np

image = cv2.imread('IMG_4244.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (0, height),
    (width/2, height/3),
    (width, height)
]

points = np.array([[0, 450], [1920, 450], [1920,1080], [0, 1080]])
print(region_of_interest_vertices)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #mask = img
    match_mask_color=(255,0,255)
    print(match_mask_color)
    #cv2.fillPoly(mask, vertices, match_mask_color)
    cv2.fillPoly(mask, [points], match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
cropped_image = region_of_interest(image,
                np.array([region_of_interest_vertices], np.int32),)
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
#cv2.imshow('edges', edges)
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),5)
#cv2.imshow('image', cropped_image)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(cropped_image)
plt.imshow(image)
plt.show()