import cv2
from MedianFlow import MedianFlow

vc = cv2.VideoCapture("media/video1.mp4")

ok, frame = vc.read()
bbox = cv2.selectROI("Tracking", frame, False)

tracker = MedianFlow()
tracker.init(frame, bbox)

#tracker = cv2.legacy.MultiTracker_create()
#tracker.add(cv2.legacy.TrackerKCF_create(), frame, bbox)
#tracker.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)
#tracker.add(cv2.legacy.TrackerMedianFlow_create(), frame, bbox)

colors = [
    (255,0,0),
    (0,255,0),
    (0,0,255)
]

while True:
    ok, frame = vc.read()
    if not ok:
        break

    ok_box, box = tracker.update(frame, level=1)
    if ok_box:
        if type(box) == tuple:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        else:
            for idx, arr in enumerate(box):
                arr = map(int, arr)
                x, y, w, h = arr
                cv2.rectangle(frame, (x, y), (x+w, y+h), colors[idx], 2)
    else:
        cv2.putText(frame, "Lost", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

vc.release()
cv2.destroyAllWindows()
