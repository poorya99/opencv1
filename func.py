import cv2
import numpy as np

img = cv2.imread("Resources/lena.png")
kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(img, 150, 200)
imgDilation = cv2.dilate(imgCanny, kernel, iterations = 1 )
imgErosion = cv2.erode(imgDilation, kernel, iterations = 1 )



cv2.imshow("Gray image", imgGray)
cv2.imshow("Blur image", imgBlur)
cv2.imshow("canny image", imgCanny)
cv2.imshow("Dilation image", imgDilation)
cv2.imshow("Eroded image", imgErosion)



cv2.waitKey(0)