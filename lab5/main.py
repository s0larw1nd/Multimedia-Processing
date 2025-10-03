import cv2
import numpy as np

video = cv2.VideoCapture("media/ЛР4_main_video.mov")

w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))

kernel = np.ones((5,5),np.float32)/25

last = None
AREA = 5

while True:
    ok, frame = video.read()
    if not(ok): 
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.filter2D(grayscale,-1,kernel)

    if last is not None:
        frame_diff = cv2.absdiff(blurred, last)
        _, frame_threshold = cv2.threshold(frame_diff,127,255,cv2.THRESH_BINARY)
        frame_contours, _ = cv2.findContours(frame_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in frame_contours:
            area = cv2.contourArea(cnt)
            
            if area > AREA:
                video_writer.write(frame)

    last = blurred

    #cv2.imshow('video', frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()