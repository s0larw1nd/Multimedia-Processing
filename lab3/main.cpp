#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

class GaussBlur {
public:
    int n;
    cv::Mat gauss;

    GaussBlur(double eps, int n_) {
        int a = n_ / 2;
        int b = n_ / 2;

        n = n_;

        gauss = cv::Mat(n_, n_, CV_64F);

        double sum = 0.0;
        for (int y = 0; y < n_; y++) {
            for (int x = 0; x < n_; x++) {
                double val = (1.0 / (2 * M_PI * eps * eps)) *
                             std::exp(-((x - a) * (x - a) + (y - b) * (y - b)) / (2 * eps * eps));
                gauss.at<double>(y, x) = val;
                sum += val;
            }
        }
        gauss /= sum;
    }

    cv::Vec3d apply_kernel(const cv::Mat &matr) {
        cv::Vec3d res(0, 0, 0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cv::Vec3b pix = matr.at<cv::Vec3b>(i, j);
                double k = gauss.at<double>(i, j);
                res[0] += pix[0] * k;
                res[1] += pix[1] * k;
                res[2] += pix[2] * k;
            }
        }
        return res;
    }

    cv::Mat blur(const cv::Mat img) {
        cv::Mat padded;
        int border = n / 2;

        cv::copyMakeBorder(img, padded, border, border, border, border, cv::BORDER_REPLICATE);

        cv::Mat new_img = cv::Mat::zeros(img.size(), img.type());

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                cv::Rect rect(x, y, n, n);
                cv::Mat pad_rect = padded(rect);
                cv::Vec3d val = apply_kernel(pad_rect);
                new_img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    cv::saturate_cast<uchar>(val[0]),
                    cv::saturate_cast<uchar>(val[1]),
                    cv::saturate_cast<uchar>(val[2])
                );
            }
        }

        return new_img;
    }

    cv::Mat getKernel() {
        return gauss;
    }
};

int main(int argc, char *argv[]) {
    int n = 3;
    if (argc > 1) { n = std::atoi(argv[1]); }
    double eps = 0.84089642;
    if (argc > 2) { eps = atof(argv[2]); }

    GaussBlur gauss(eps, n);

    cv::Mat img = cv::imread("./img.png");

    cv::imshow("Window_orig", img);

    cv::Mat my_blur = gauss.blur(img);
    cv::imshow("Window blur", my_blur);

    cv::Mat cv2_blur;
    cv::GaussianBlur(img, cv2_blur, cv::Size(n, n), eps);
    cv::imshow("Window CV2", cv2_blur);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
