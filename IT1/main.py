import cv2
from MedianFlow import MedianFlow
from KCF import KCF

n = 4

vc = cv2.VideoCapture(f"media/video{n}.mp4")

ok, frame = vc.read()
bbox = cv2.selectROI("Tracking", frame, False)

tracker = cv2.legacy.MultiTracker_create()
tracker.add(cv2.legacy.TrackerKCF_create(), frame, bbox)
tracker.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)
tracker.add(cv2.legacy.TrackerTLD_create(), frame, bbox)

colors = [
    (255,0,0),
    (0,255,0),
    (0,0,255)
]

w = int(frame.shape[1])
h = int(frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'XVID')

'''
writers = [
    cv2.VideoWriter(f"output{n}_KCF.mov", fourcc, 25, (w, h)),
    cv2.VideoWriter(f"output{n}_CSRT.mov", fourcc, 25, (w, h)),
    cv2.VideoWriter(f"output{n}_TLD.mov", fourcc, 25, (w, h))
]
'''

while True:
    ok, frame = vc.read()
    if not ok:
        break

    ok_box, box = tracker.update(frame)
    if ok_box:
        if type(box) == tuple:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        else:
            #orig_frame = frame.copy()
            for idx, arr in enumerate(box):
                arr = map(int, arr)
                x, y, w, h = arr
                cv2.rectangle(frame, (x, y), (x+w, y+h), colors[idx], 2)

                #writeframe = orig_frame.copy()
                #cv2.rectangle(writeframe, (x, y), (x+w, y+h), colors[idx], 2)

                #writers[idx].write(writeframe)
    else:
        cv2.putText(frame, "Lost", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

vc.release()
cv2.destroyAllWindows()
