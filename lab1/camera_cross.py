import cv2
import numpy as np

cap = cv2.VideoCapture("media/sample.mp4")

while True:
    ok, img = cap.read()
    if not ok:
        break

    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    length = 300
    width = 40
    thickness = 2

    color = (int(255*(np.argmax(img[center_x][center_y])==0)), int(255*(np.argmax(img[center_x][center_y])==1)), int(255*(np.argmax(img[center_x][center_y])==2)))

    cv2.rectangle(img, (center_x-int(length/2), center_y-int(width/2)), (center_x+int(length/2), center_y+int(width/2)), color, -1)
    cv2.rectangle(img, (center_x-int(width/2), center_y-int(length/2)), (center_x+int(width/2), center_y+int(length/2)), color, -1)

    cv2.imshow("Camera with cross", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()