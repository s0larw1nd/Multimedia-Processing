import cv2 as cv
import numpy as np

low_H = 0
low_S = 100
low_V = 100

high_H = 10
high_S = 255
high_V = 255

kernel = 10

cv.namedWindow('Video Capture')
cv.namedWindow('Object Detection')
cv.namedWindow('Open')

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        break   
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.bitwise_or(cv.inRange(frame_HSV, (0, low_S, low_V), (10, high_S, high_V)),
                                    cv.inRange(frame_HSV, (160, low_S, low_V), (180, high_S, high_V)))

    kernel_matr = np.ones((kernel, kernel), np.uint8)

    img_temp = cv.erode(frame_threshold, kernel_matr, iterations=1)
    img_morph = cv.dilate(img_temp, kernel_matr, iterations=1)

    m00 = 0
    m10 = 0
    m01 = 0

    m20 = 0
    m11 = 0
    m02 = 0
    
    for y in range(len(img_morph)):
        for x in range(len(img_morph[y])):
            if (img_morph[y][x]>0):
                m00 += 1
                m10 += x
                m01 += y

                m20 += x*x
                m02 += y*y
                m11 += x*y

    try:
        x = int(m10/m00)
        y = int(m01/m00)
        mt = np.array([[(m20-x*m10)/m00, (m11-x*m01)/m00],[(m11-x*m01)/m00, (m02-y*m01)/m00]])
        eigenvalues, _ = np.linalg.eig(mt)

        w, h = 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1])
    except Exception:
        pass

    try:
        cv.circle(frame, (x, y), 0, (0, 0, 0), 10)
        cv.rectangle(frame, (int(max(x-w/2,10)), int(max(y-h/2,10))), (int(max(x+w/2,10)), int(max(y+h/2,10))), (0, 0, 0), 5)
    except Exception:
        pass
    
    cv.imshow('Video Capture', frame)
    cv.imshow('Object Detection', frame_threshold)
    cv.imshow('Open', img_morph)
    
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break