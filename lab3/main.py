import numpy as np
import cv2

class GaussBlur:
    def __init__(self, eps, n):
        self.n = n
        a = self.n//2
        b = self.n//2
        self.gauss = []
        for y in range(n):
            self.gauss.append([])
            for x in range(n):
                self.gauss[-1].append(1/(2*np.pi*eps*eps)*np.e**(-((x-a)**2+(y-b)**2)/(2*eps*eps)))
        self.gauss = np.array(self.gauss)
        self.gauss /= np.sum(self.gauss)

    def apply_kernel(self, matr):
        res = np.zeros(3)
        for i in range(len(matr)):
            for j in range(len(matr[i])):
                res += matr[i][j] * self.gauss[i][j]
        return res

    def blur(self, img):
        img_temp = img

        new_img = np.zeros_like(img)

        for _ in range(self.n//2):
            img_temp = np.insert(img_temp, 0, img_temp[0], axis=0)
            img_temp = np.insert(img_temp, -1, img_temp[-1], axis=0)
        for _ in range(self.n//2):
            img_temp = np.insert(img_temp, 0, img_temp[:,0], axis=1)
            img_temp = np.insert(img_temp, -1, img_temp[:,-1], axis=1)

        for y in range(len(img)):
            ty = y+self.n//2
            for x in range(len(img[y])):
                tx = x+self.n//2
                new_img[y, x] = self.apply_kernel(img_temp[ty-(self.n-1)//2:ty+(self.n-1)//2+1, tx-(self.n-1)//2:tx+(self.n-1)//2+1])

        return new_img

gauss = GaussBlur(15, 3)

img = cv2.imread("media/img.png")
cv2.imshow("Window_orig", img)
cv2.imshow("Window blur", gauss.blur(img))

cv2_img = cv2.filter2D(src=img, ddepth=-1, kernel=gauss.gauss)
cv2.imshow("Window CV2", cv2_img)

cv2.waitKey(0)
cv2.destroyAllWindows()