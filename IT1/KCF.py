import cv2
import numpy as np

class KCF:
    @classmethod
    def gaussian_label(cls, size, sigma):
        h, w = size
        xs = np.arange(w) - w / 2
        ys = np.arange(h) - h / 2
        xx, yy = np.meshgrid(xs, ys)
        y = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        y /= np.max(y)
        return y
    
    @classmethod
    def kernel_correlation(cls, x, z, sigma=2):
        x = x - np.mean(x)
        z = z - np.mean(z)
        xf = np.fft.fft2(x)
        zf = np.fft.fft2(z)
        xx = np.sum(x ** 2)
        zz = np.sum(z ** 2)
        xz = np.real(np.fft.ifft2(xf * np.conj(zf)))
        k = np.exp(-1 / (sigma ** 2) * np.maximum(0, (xx + zz - 2 * xz) / x.size))
        return k
    
    @classmethod
    def extract_features(cls, frame, bbox, padding=1.5):
        frame = cv2.GaussianBlur(frame, (3,3), 0)

        cx, cy = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
        w = int(bbox[2] * padding)
        h = int(bbox[3] * padding)
        x1 = max(cx - w // 2, 0)
        y1 = max(cy - h // 2, 0)
        x2 = min(cx + w // 2, frame.shape[1])
        y2 = min(cy + h // 2, frame.shape[0])

        window = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        gray = (gray - gray.mean()) / (gray.std() + 1e-5)
        hann = np.outer(np.hanning(64), np.hanning(64))

        return gray * hann

    def init(self, frame, bbox):
        features_x = KCF.extract_features(frame, bbox)

        y = KCF.gaussian_label((64, 64), sigma=2)
        self.y_fft = np.fft.fft2(y)

        kxx = KCF.kernel_correlation(features_x, features_x)
        kxx_fft = np.fft.fft2(kxx)

        alpha = self.y_fft / (kxx_fft + 0.001)

        self.model_x = features_x
        self.model_a = alpha
        self.prev_bbox = bbox

    def update(self, next_frame, lambd=0.01, n=0.05):
        features_z = KCF.extract_features(next_frame, self.prev_bbox)

        kxz = KCF.kernel_correlation(features_z, self.model_x)
        kxz_fft = np.fft.fft2(kxz)

        response = np.real(np.fft.ifft2(self.model_a * kxz_fft))

        kzz = KCF.kernel_correlation(features_z, features_z)
        kzz_fft = np.fft.fft2(kzz)
        alpha_new = self.y_fft / (kzz_fft + lambd)

        self.model_a = (1 - n) * self.model_a + n * alpha_new
        self.model_x = (1 - n) * self.model_x + n * features_z

        max_idx = np.unravel_index(np.argmax(response), response.shape)
        scale_x = self.prev_bbox[2] / 64.0
        scale_y = self.prev_bbox[3] / 64.0
        dx = (max_idx[1] - response.shape[1] // 2) * scale_x
        dy = (max_idx[0] - response.shape[0] // 2) * scale_y

        if abs(dx) > next_frame.shape[0]/4: dx = 0
        if abs(dy) > next_frame.shape[1]/4: dy = 0

        new_center_x = self.prev_bbox[0] + self.prev_bbox[2] // 2 + dx
        new_center_y = self.prev_bbox[1] + self.prev_bbox[3] // 2 + dy

        return True, (int(new_center_x - self.prev_bbox[2] // 2), int(new_center_y - self.prev_bbox[3] // 2), self.prev_bbox[2], self.prev_bbox[3])