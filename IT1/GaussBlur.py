import numpy as np

class GaussBlur:
    def __init__(self, eps=0.84089642, n=3):
        self.n = n
        a = self.n//2+1
        b = self.n//2+1
        self.gauss = []
        for y in range(n):
            self.gauss.append([])
            for x in range(n):
                self.gauss[-1].append(1/(2*np.pi*eps*eps)*np.exp(-((x-a)**2+(y-b)**2)/(2*eps*eps)))
        self.gauss = np.array(self.gauss)
        self.gauss /= np.sum(self.gauss)

    def apply_kernel(self, matr):
        return np.sum(matr * self.gauss)

    def blur(self, img):
        img_temp = img.copy()

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