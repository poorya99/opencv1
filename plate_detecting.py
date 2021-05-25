import cv2
import numpy as np


plateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.XML")
minArea = 100

img = cv2.imread("Resources/plate2.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
numberPlate = plateCascade.detectMultiScale(imgGray, 1.1, 4)

for (x, y, w, h) in numberPlate:
    print(x, y, w, h)
    area = w*h
    if area > minArea:
        cv2. rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Number Plate", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        imgRoi = img[y:y+h, x:x+w]





imshape = img.shape
# print(imshape)
# img = cv2.resize(img, (imshape[1]//3, imshape[0]//3))

# cv2.imshow("image", imgRoi)
cv2.imshow("image", img)

cv2.waitKey(0)
