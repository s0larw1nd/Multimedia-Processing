import cv2
import numpy as np
from GaussBlur import GaussBlur

class MedianFlow:
    def __init__(self):
        self.DOTS_NUM = 5
        self.WINDOW_SIZE = 15

        self.Gx = np.array([[-1,0,1],
               [-2,0,2],
               [-1,0,1]])

        self.Gy = np.array([[-1,-2,-1],
               [ 0, 0, 0],
               [ 1, 2, 1]])
        
        self.last_frame = np.array([])
        self.bbox = (0,0,0,0)

        self.gauss = GaussBlur(0.84089642, 3)

    def init(self, frame, bbox):
        self.last_frame = frame
        self.bbox = bbox

    def Lucas_Kanade(self, img, img1, pairs, window_size=10, max_iter=3, eps=0.001):
        new_pairs = []
        half = window_size // 2
        h, w = img.shape

        Ix = np.zeros(img.shape)
        Iy = np.zeros(img.shape)
        img_temp = img.copy()

        img_temp = np.pad(img, pad_width=1, mode='edge')

        min_x = min([p[0] for p in pairs])
        min_y = min([p[1] for p in pairs])
        max_x = max([p[0] for p in pairs])
        max_y = max([p[1] for p in pairs])

        for y in range(int(min_y-half),int(max_y+half+1)):
            yt = y+1
            for x in range(int(min_x-half),int(max_x+half+1)):
                xt = x+1

                if yt-1>=0 and yt+2<len(img_temp) and xt-1>=0 and xt+2<len(img_temp[yt]): 
                    patch = img_temp[yt-1:yt+2, xt-1:xt+2]
                else: 
                    raise ValueError("Ошибка при вычислении градиентов")

                Ix[y][x] = np.sum(patch * self.Gx)
                Iy[y][x] = np.sum(patch * self.Gy)

        It = img1 - img

        for x, y in pairs:
            u, v = 0, 0

            if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
                new_pairs.append((u, v))
                continue

            Ix_win = Ix[int(y-half):int(y+half+1), int(x-half):int(x+half+1)].reshape(-1)
            Iy_win = Iy[int(y-half):int(y+half+1), int(x-half):int(x+half+1)].reshape(-1)

            A11 = np.sum(Ix_win * Ix_win)
            A12 = np.sum(Ix_win * Iy_win)
            A22 = np.sum(Iy_win * Iy_win)

            A = np.array([[A11, A12], 
                        [A12, A22]])
            
            for _ in range(max_iter):
                It_win = It[int(y+v-half):int(y+v+half+1), int(x+u-half):int(x+u+half+1)].reshape(-1)

                b1 = np.sum(Ix_win * It_win)
                b2 = np.sum(Iy_win * It_win)
                        
                b = np.array([-b1, -b2])

                if np.linalg.det(A) == 0:
                    new_pairs.append((x, y))
                    continue

                uv = np.linalg.inv(A) @ b
                delta_u, delta_v = uv.ravel()

                u += delta_u
                v += delta_v

                if np.sqrt(delta_u**2 + delta_v**2) < eps: break

            new_pairs.append((u, v))

        return new_pairs
    
    def Lucas_KanadeCV(self, img: np.ndarray, img2: np.ndarray, pairs: list[tuple[int, int]]):
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        points = np.array(pairs, dtype=np.float32).reshape(-1, 1, 2)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, points, None)
        valid_new_points = new_points[status.flatten() == 1]
        valid_new_points_list = [tuple(map(int, pt)) for pt in valid_new_points.reshape(-1, 2)]
        return valid_new_points_list
    
    def euclid(self, x, y):
        v = 0
        for i,j in zip(x,y):
            v += (i-j)**2
        return np.sqrt(v)

    def update(self, frame, level=5, use_orig=True):
        if self.last_frame.size == 0:
            self.last_frame = frame

        else:
            img = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            img_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if level>1:
                if not(use_orig): level += 1
                #pyramid_t = [self.gauss.blur(img)]
                #pyramid_t1 = [self.gauss.blur(img_next)]

                #pyramid_t = [cv2.GaussianBlur(img, (3, 3), 1)]
                #pyramid_t1 = [cv2.GaussianBlur(img_next, (3, 3), 1)]

                pyramid_t = [cv2.filter2D(src=img, ddepth=-1, kernel=self.gauss.gauss)]
                pyramid_t1 = [cv2.filter2D(src=img_next, ddepth=-1, kernel=self.gauss.gauss)]

                for l in range(level-1):
                    #pyramid_t.append(self.gauss.blur(pyramid_t[-1])[::2, ::2])
                    #pyramid_t1.append(self.gauss.blur(pyramid_t1[-1])[::2, ::2])

                    #pyramid_t.append(cv2.GaussianBlur(pyramid_t[-1], (3, 3), 1)[::2, ::2])
                    #pyramid_t1.append(cv2.GaussianBlur(pyramid_t1[-1], (3, 3), 1)[::2, ::2])

                    pyramid_t.append(cv2.filter2D(src=pyramid_t[-1], ddepth=-1, kernel=self.gauss.gauss)[::2, ::2])
                    pyramid_t1.append(cv2.filter2D(src=pyramid_t1[-1], ddepth=-1, kernel=self.gauss.gauss)[::2, ::2])

            offset_x = self.bbox[2] // self.DOTS_NUM
            offset_y = self.bbox[3] // self.DOTS_NUM

            pairs = [(self.bbox[0]+offset_x*j, self.bbox[1]+offset_y*i) for i in range(self.DOTS_NUM) for j in range(self.DOTS_NUM)]

            if level == 1:
                forward_pairs = self.Lucas_Kanade(img, img_next, pairs)
                backward_pairs = self.Lucas_Kanade(img_next, img, forward_pairs)
            else:
                forward_pairs = pairs
                end = 0
                if use_orig: end = -1
                for l in range(len(pyramid_t)-1, end, -1):
                    scale = 2**l
                    p_scaled = [(x/scale, y/scale) for x,y in forward_pairs]
                    deltas = self.Lucas_Kanade(pyramid_t[l], pyramid_t1[l], p_scaled)
                    #deltas, status, err = cv2.calcOpticalFlowPyrLK(np.array(pyramid_t[l]), np.array(pyramid_t1[l]), np.array(p_scaled), None)
                    forward_pairs = [(x + (dx * scale), y + (dy * scale)) for (x, y), (dx, dy) in zip(forward_pairs, deltas)]

                backward_pairs = forward_pairs.copy()
                for l in range(len(pyramid_t)-1, end, -1):
                    scale = 2**l
                    p_scaled = [(x/scale, y/scale) for x,y in backward_pairs]
                    deltas = self.Lucas_Kanade(pyramid_t1[l], pyramid_t[l], p_scaled)
                    #deltas, status, err = cv2.calcOpticalFlowPyrLK(np.array(pyramid_t1[l]), np.array(pyramid_t[l]), np.array(p_scaled), None)
                    forward_pairs = [(x + (dx * scale), y + (dy * scale)) for (x, y), (dx, dy) in zip(backward_pairs, deltas)]

            ei = []
            for i, e in enumerate(zip(pairs, backward_pairs)):
                p,bp = e    
                ei.append((i, float(self.euclid(p,bp))))
            m = np.median([i[1] for i in ei])
            median_err = np.median([e[1] for e in ei])
            mad = np.median([abs(e[1] - median_err) for e in ei]) + 1e-6
            ei = [e for e in ei if e[1] < median_err + 1.5 * mad]
            eii = [i[0] for i in ei]

            dx = [forward_pairs[i][0]-pairs[i][0] for i in eii]
            dy = [forward_pairs[i][1]-pairs[i][1] for i in eii]
            dmed = (np.median(dx), np.median(dy))

            r = []
            for i in range(len(eii)):
                for j in range(i, len(eii)):
                    r1 = self.euclid(forward_pairs[eii[i]],forward_pairs[eii[j]])
                    r2 = self.euclid(pairs[eii[i]],pairs[eii[j]])
                    if r2 != 0: r.append(r1/r2)
            s = np.median(r)
            s = np.clip(s, 0.9, 1.1)

            self.last_frame = frame
            
            self.bbox = (int(self.bbox[0]+dmed[0]),int(self.bbox[1]+dmed[1]), int(s*self.bbox[2]), int(s*self.bbox[3]))
            self.last_frame = frame

            return True, self.bbox