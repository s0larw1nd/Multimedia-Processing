#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

float apply_kernel(const Mat& matr, const Mat& ker) {
    float res = 0.0f;
    for (int i = 0; i < ker.rows; i++) {
        for (int j = 0; j < ker.cols; j++) {
            res += matr.at<uchar>(i, j) * ker.at<float>(i, j);
        }
    }
    return res;
}

int quantize_angle(float angle_deg) {
    if (0 <= angle_deg && angle_deg < 22.5) return 2;
    else if (22.5 <= angle_deg && angle_deg < 67.5) return 1;
    else if (67.5 <= angle_deg && angle_deg < 112.5) return 0;
    else if (112.5 <= angle_deg && angle_deg < 157.5) return 7;
    else if (157.5 <= angle_deg && angle_deg < 202.5) return 6;
    else if (202.5 <= angle_deg && angle_deg < 247.5) return 5;
    else if (247.5 <= angle_deg && angle_deg < 292.5) return 4;
    else if (292.5 <= angle_deg && angle_deg < 337.5) return 3;
    else return 2;
}

int main() {
    Mat img = imread("./img.png", IMREAD_GRAYSCALE);

    Mat kernel = Mat::ones(5, 5, CV_32F) / 25.0;
    Mat dst;
    filter2D(img, dst, -1, kernel);

    Mat Gx = (Mat_<float>(3,3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    );
    Mat Gy = (Mat_<float>(3,3) <<
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    );

    Mat new_img = Mat::zeros(img.size(), CV_32F);
    Mat grads = Mat::zeros(img.size(), CV_32F);

    float mxl = -1.0f;

    for (int y = 1; y < dst.rows - 1; y++) {
        for (int x = 1; x < dst.cols - 1; x++) {
            Mat roi = img(Rect(x-1, y-1, 3, 3));
            float gx = apply_kernel(roi, Gx);
            float gy = apply_kernel(roi, Gy);
            float l = sqrt(gx * gx + gy * gy);
            grads.at<float>(y, x) = l;
            if (l > mxl) mxl = l;
        }
    }

    for (int y = 1; y < dst.rows - 1; y++) {
        for (int x = 1; x < dst.cols - 1; x++) {
            Mat roi = img(Rect(x-1, y-1, 3, 3));
            float gx = apply_kernel(roi, Gx);
            float gy = apply_kernel(roi, Gy);

            float angle = atan2(gy, gx) * 180.0 / CV_PI;
            int angle_round = quantize_angle(angle);

            float val = grads.at<float>(y, x);
            if ((angle_round == 2 || angle_round == 6) && val > max(grads.at<float>(y, x-1), grads.at<float>(y, x+1)))
                new_img.at<float>(y, x) = 255;
            else if ((angle_round == 0 || angle_round == 4) && val > max(grads.at<float>(y-1, x), grads.at<float>(y+1, x)))
                new_img.at<float>(y, x) = 255;
            else if ((angle_round == 3 || angle_round == 7) && val > max(grads.at<float>(y-1, x-1), grads.at<float>(y+1, x+1)))
                new_img.at<float>(y, x) = 255;
            else if ((angle_round == 1 || angle_round == 5) && val > max(grads.at<float>(y-1, x+1), grads.at<float>(y+1, x-1)))
                new_img.at<float>(y, x) = 255;
        }
    }

    float low_level = mxl / 25.0f;
    float high_level = mxl / 10.0f;

    Mat borders = Mat::zeros(img.size(), CV_8U);

    for (int y = 1; y < new_img.rows - 1; y++) {
        for (int x = 1; x < new_img.cols - 1; x++) {
            float val = new_img.at<float>(y, x);
            if (val >= high_level) {
                borders.at<uchar>(y, x) = 255;
            } else if (val >= low_level && val < high_level) {
                bool strong_neighbor = false;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (new_img.at<float>(y+dy, x+dx) >= high_level)
                            strong_neighbor = true;
                    }
                }
                if (strong_neighbor) borders.at<uchar>(y, x) = 255;
            }
        }
    }

    imshow("Display window: Original", img);
    imshow("Display window: Gauss", dst);
    imshow("Display window: Borders", borders);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
