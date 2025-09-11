import cv2

img = cv2.imread("media/img.jpg" #, cv2.IMREAD_UNCHANGED
                                #, cv2.IMREAD_GRAYSCALE
                                #, cv2.IMREAD_COLOR_RGB
                                )


cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
#cv2.setWindowProperty('Display window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.setWindowProperty('Display window', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

cv2.imshow('Display window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()