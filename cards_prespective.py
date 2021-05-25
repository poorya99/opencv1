import cv2
import numpy as np

img = cv2.imread("Resources/cards.jpg")

width , height = 250,350
pst1 = np.float32([[219, 150], [348, 150], [193, 300], [336, 300]])
pst2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pst1,pst2)
imageOutput = cv2.warpPerspective(img, matrix, (width, height))


cv2.imshow("image", img)
cv2.imshow("output", imageOutput)


cv2.waitKey(0)
