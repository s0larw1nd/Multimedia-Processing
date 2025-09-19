import cv2

def readIPWriteTOFile():
    video = cv2.VideoCapture(r"media/sample.mp4", cv2.CAP_ANY)

    ok, img = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))

    video.set(cv2.CAP_PROP_FRAME_WIDTH, w // 8)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, h // 8)

    while (True):
        ok, img = video.read()
        if not(ok): 
            break
        
        new_img = cv2.resize(img, (int(w/2), int(h/2)))
        #new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('video', new_img)
        video_writer.write(img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

readIPWriteTOFile()