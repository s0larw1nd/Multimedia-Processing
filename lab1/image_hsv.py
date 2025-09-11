import cv2

img = cv2.imread("media/img.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', img)

cv2.namedWindow('Image HSV', cv2.WINDOW_NORMAL)
cv2.imshow('Image HSV', img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()