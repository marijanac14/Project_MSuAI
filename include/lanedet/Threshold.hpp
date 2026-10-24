#pragma once

#include <opencv2/opencv.hpp>

namespace lanedet {

// Thresholds for building the binary lane mask. Colour picks up the markings
// directly (white and yellow); the gradient catches line edges the colour
// masks miss. All of it runs in HLS, where white is simply high lightness and
// yellow has a distinctive hue regardless of how bright the road is.
struct ThresholdParams {
    int whiteLightnessMin = 200;

    // HLS order is (H, L, S); yellow sits in a narrow hue band.
    cv::Scalar yellowLow{15, 30, 100};
    cv::Scalar yellowHigh{35, 204, 255};

    int sobelKernel = 3;
    int gradientMin = 30;
    int gradientMax = 255;
    bool useGradient = true;
};

// The individual masks, kept separately so we can tune them and show each one
// in the writeup.
struct ThresholdMasks {
    cv::Mat white;
    cv::Mat yellow;
    cv::Mat gradient;
    cv::Mat combined;
};

cv::Mat buildBinary(const cv::Mat& bgr, const ThresholdParams& params = {});

ThresholdMasks buildBinaryMasks(const cv::Mat& bgr,
                                const ThresholdParams& params = {});

} // namespace lanedet
