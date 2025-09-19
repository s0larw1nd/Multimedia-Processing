import cv2 as cv

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 100
low_V = 100
high_H = 10 #0-10 AND 160-180
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
window_morph_name = 'Dilation'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
kernel = 1

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_dilation_trackbar(val):
    global kernel
    kernel = val
    cv.setTrackbarPos(title_trackbar_kernel_size, window_morph_name, kernel)

cap = cv.VideoCapture(0)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.namedWindow(window_morph_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

cv.createTrackbar(title_trackbar_kernel_size, window_morph_name, 1, 15, on_dilation_trackbar)

import numpy as np
while True:
    ret, frame = cap.read()
    if frame is None:
        break   
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

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
        cv.circle(frame, (x, y), 0, (0, 0, 255), 10)
        cv.rectangle(frame, (int(max(x-w/2,10)), int(max(y-h/2,10))), (int(max(x+w/2,10)), int(max(y+h/2,10))), (0, 0, 255), 5)
    except Exception:
        pass
    
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    cv.imshow(window_morph_name, img_morph)
    
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break