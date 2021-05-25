import cv2
import numpy as np

img = cv2.imread("Resources/lena.png")
imshape = img.shape
halfSize1 = int(imshape[0]/2)
halfSize2 = int(imshape[1]/2)
#print(halfSize)

imgResize = cv2.resize(img, (halfSize1, halfSize2))

imgCropped = img[0:200,200:500] #height , width


cv2.imshow("image", img)
cv2.imshow("image Resize", imgResize)
cv2.imshow("image cropeed", imgCropped)

cv2.waitKey(0)
