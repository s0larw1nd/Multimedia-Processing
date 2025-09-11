import cv2

def readIPWriteTOFile():
    video = cv2.VideoCapture(0, cv2.CAP_ANY)

    video.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    video.set(cv2.CAP_PROP_CONTRAST, 0.5)
    video.set(cv2.CAP_PROP_SATURATION, 0.5)
    video.set(cv2.CAP_PROP_FPS, 30)

    ok, img = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))

    while (True):
        ok, img = video.read()
        if not(ok): 
            break
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

readIPWriteTOFile()