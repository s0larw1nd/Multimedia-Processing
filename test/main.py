import cv2
import numpy as np
import mss
import time
import os

os.makedirs("frames", exist_ok=True)

fps = 10
delay = 1 / fps
frame_id = 0

with mss.mss() as sct:
    print("Запись начата. Нажмите Ctrl+C для остановки.")
    
    try:
        while True:
            start = time.time()

            img = sct.grab(sct.monitors[1])
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            filename = f"frames/frame_{frame_id:06d}.png"
            cv2.imwrite(filename, frame)

            frame_id += 1

            elapsed = time.time() - start
            if elapsed < delay:
                time.sleep(delay - elapsed)

    except KeyboardInterrupt:
        print("\nОстановка записи.")
