import cv2
import numpy as np
# naive version
points = np.array([[0, 450], [1920, 450], [1920,1080], [0, 1080]])
# print(region_of_interest_vertices)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #mask = img
    match_mask_color=(255,0,255)
    print(match_mask_color)
    #cv2.fillPoly(mask, vertices, match_mask_color)
    cv2.fillPoly(mask, [points], match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def process(image):
    copy_image = image


    cropped_image = region_of_interest(image,
                                       np.array([points], np.int32), )
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imshow('edges', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    print(lines)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)



    ptD = [700, 500]
    ptC = [1200, 486]
    ptA = [0, 800]
    ptB = [2000, 966]

    wi, hi = 1500, 1500
    # cv2.imshow('image', copy_image)
    pts1 = np.float32([ptD, ptC, ptA, ptB])
    pts2 = np.float32([[0, 0], [wi, 0], [0, hi], [wi, hi]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (wi, hi))


    font = cv2.FONT_HERSHEY_SIMPLEX
    st = 1500
    cv2.line(result, (350, st), (150, st), (255, 0, 0), 50)
    cv2.line(result, (1400, st), (1200, st), (255, 0, 0), 50)

    cv2.line(result, (350, st - 340), (150, st - 340), (255, 0, 0), 50)
    cv2.line(result, (1400, st - 340), (1200, st - 340), (255, 0, 0), 50)
    cv2.putText(result, '1 M', (600, st - 340), font, 5,
                (255, 0, 0), 20, cv2.LINE_AA, False)

    cv2.line(result, (350, st - 340 * 2), (150, st - 340 * 2), (255, 0, 0), 50)
    cv2.line(result, (1400, st - 340 * 2), (1200, st - 340 * 2), (255, 0, 0), 50)
    cv2.putText(result, '2 M', (600, st - 340 * 2), font, 5,
                (255, 0, 0), 20, cv2.LINE_AA, False)

    cv2.line(result, (350, st - 340 * 3), (150, st - 340 * 3), (255, 0, 0), 50)
    cv2.line(result, (1400, st - 340 * 3), (1200, st - 340 * 3), (255, 0, 0), 50)
    cv2.putText(result, '3 M', (600, st - 340 * 3), font, 5,
                (255, 0, 0), 20, cv2.LINE_AA, False)
    # print(np.linalg.inv(result))
    # plt.imshow(result)
    print(result)
    result = np.array(result)
    print('*' * 100)
    IMAGE_H = 1080
    IMAGE_W = 1920

    matrix = (np.linalg.inv(matrix))
    result1 = cv2.warpPerspective(result, matrix, (IMAGE_W, IMAGE_H))
    output = cv2.bitwise_or(copy_image, result1)


    return result1

def output(img,mark):
    tod = 10
    ptD = [700+tod, 500+tod]
    ptC = [1200+tod, 486+tod]
    ptA = [0, 800]
    ptB = [2000-tod, 966-tod]

    sorted_pts = np.float32([ptC, ptD, ptA, ptB])
    mask = np.zeros(mark.shape, dtype=np.uint8)

    roi_corners = np.int32(sorted_pts)
    print(sorted_pts)

    cv2.fillConvexPoly(mask, roi_corners, (255, 255, 255))
    mask = cv2.bitwise_not(mask)
    # cv2.imshow('Fused Image', mask)
    masked_image = cv2.bitwise_and(img, mask)
    output = cv2.bitwise_or(mark, masked_image)
    cv2.imshow('image', output)

# cap = cv2.VideoCapture('./image/IMG_4244.MOV')
# image = cv2.imread('IMG_4244.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cap = cv2.VideoCapture('./image/IMG_4244.mp4')
cap1 = cv2.VideoCapture('./image/IMG_4244.mp4')
success, img = cap.read()
success1, img1 = cap.read()
while success:
    output(img1, process(img))
    success, img = cap.read()
    success1, img1 = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()