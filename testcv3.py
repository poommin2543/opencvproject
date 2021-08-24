import matplotlib.pylab as plt
import cv2
import numpy as np

image = cv2.imread('road.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

'''region_of_interest_vertices = [
    (0, height),
    (width/2, height/3),
    (width, height)
]'''
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
#print(cropped_image)
plt.imshow(cropped_image)
plt.show()
