import time
import pyautogui
import numpy as np
import cv2
from ultralytics import YOLO
from mss import mss

import sys
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget

class Overlay(QWidget):
    def __init__(self):
        super().__init__()

        self.rectangles = []     # список прямоугольников (x, y, w, h)

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint |
                            Qt.WindowStaysOnTopHint |
                            Qt.Tool)

        self.showFullScreen()

    def set_rectangles(self, rect_list):
        """Получает новые координаты фигур и перерисовывает."""
        self.rectangles = rect_list
        self.update()        # триггер перерисовки

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.red, 4)
        painter.setPen(pen)

        for x, y, w, h in self.rectangles:
            painter.drawRect(x, y, w, h)

model = YOLO("runs/detect/train/weights/best.pt")
#print(model.names)

sct = mss()
monitor = sct.monitors[1]

SCREEN_W = monitor["width"]
SCREEN_H = monitor["height"]

sx = SCREEN_W // 2
sy = SCREEN_H // 2

# {0: 'counter', 1: 'counter_head', 2: 'ter', 3: 'ter_head'}
TARGET_CLASS = 0
MIN_CONF = 0.5

FRAMES = 3
COEF = 0.1

def get_center_of_first_box(result):
    if len(result.boxes) == 0:
        return None
    
    boxes = []
    
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        diff = np.sqrt(np.pow(sx - cx,2) + np.pow(sy-cy,2))
        
        if int(b.cls) == TARGET_CLASS and float(b.conf) >= MIN_CONF: boxes.append((diff,cx,cy,(x1, y1, x2, y2),float(b.conf)))
    
    if boxes == []: return None
    
    top = sorted(boxes, key=lambda x: x[0])[0]
    #print(top)
    return top[1], top[2], top[3]

def k(x, C=10):
    return 1 - np.exp(-x/C)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = Overlay()
    
    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        res = model.predict(frame, save=False, verbose=False)[0]
        
        center = get_center_of_first_box(res)
        if center is None:
            continue

        cx, cy, coords = center

        dx = cx - sx
        dy = cy - sy
        
        print(cx, sx, dx, cy, sy, dy)

        move_x = dx / (FRAMES)
        move_y = dy / (FRAMES)
        
        pyautogui.moveRel(move_x, move_y, duration=0)
        
        overlay.set_rectangles([coords])

        time.sleep(min(1/FRAMES, 0.00001))