import cv2
import numpy as np

img = cv2.imread("media/img.png", cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

def apply_kernel(matr, ker):
    res = 0
    for i in range(len(matr)):
        for j in range(len(matr[i])):
            res += matr[i][j] * ker[i][j]
    return res

Gx = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

Gy = np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
])

def quantize_angle(angle_deg):
    if 0 <= angle_deg < 22.5:
        return 2
    elif 22.5 <= angle_deg < 67.5:
        return 1
    elif 67.5 <= angle_deg < 112.5:
        return 0
    elif 112.5 <= angle_deg < 157.5:
        return 7
    elif 157.5 <= angle_deg < 202.5:
        return 6
    elif 202.5 <= angle_deg < 247.5:
        return 5
    elif 247.5 <= angle_deg < 292.5:
        return 4
    elif 292.5 <= angle_deg < 337.5:
        return 3
    else:
        return 2
    
new_img = np.zeros(img.shape)
grads = np.zeros(img.shape)

mxl = -1
for y in range(1, len(dst)-1):
    for x in range(1, len(dst[y])-1):
        gx = apply_kernel(img[y-1:y+2,x-1:x+2], Gx)
        gy = apply_kernel(img[y-1:y+2,x-1:x+2], Gy)

        l = np.sqrt(gx**2+gy**2)
        grads[y,x] = l
        mxl = max(mxl,l)

for y in range(1, len(dst)-1):
    for x in range(1, len(dst[y])-1):
        gx = apply_kernel(img[y-1:y+2,x-1:x+2], Gx)
        gy = apply_kernel(img[y-1:y+2,x-1:x+2], Gy)

        angle = np.rad2deg(np.arctan2(gy, gx))
        angle_round = quantize_angle(angle)

        if angle_round in [2,6] and grads[y,x]>max(grads[y,x-1],grads[y,x+1]):
            new_img[y,x] = 255
        elif angle_round in [0,4] and grads[y,x]>max(grads[y-1,x],grads[y+1,x]):
            new_img[y,x] = 255
        elif angle_round in [3,7] and grads[y,x]>max(grads[y-1,x-1],grads[y+1,x+1]):
            new_img[y,x] = 255
        elif angle_round in [1,5] and grads[y,x]>max(grads[y-1,x+1],grads[y+1,x-1]):
            new_img[y,x] = 255

low_level = mxl // 25
high_level = mxl // 10

borders = np.zeros(img.shape)

for y in range(1, len(new_img)-1):
    for x in range(1, len(new_img[y])-1):
        if new_img[y,x] >= high_level: borders[y,x] = 255
        elif low_level <= new_img[y,x] < high_level:
            if any(x >= high_level for x in [
                new_img[y-1,x],
                new_img[y+1,x],
                new_img[y,x-1],
                new_img[y,x+1],
                new_img[y-1,x-1],
                new_img[y-1,x+1],
                new_img[y+1,x-1],
                new_img[y+1,x+1]
            ]): borders[y,x] = 255

cv2.imshow('Display window1', dst)
cv2.imshow('Display window2', borders)

cv2.waitKey(0)
cv2.destroyAllWindows()