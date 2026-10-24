#include "lanedet/Threshold.hpp"

#include <vector>

namespace lanedet {

ThresholdMasks buildBinaryMasks(const cv::Mat& bgr,
                                const ThresholdParams& params) {
    cv::Mat hls;
    cv::cvtColor(bgr, hls, cv::COLOR_BGR2HLS);

    ThresholdMasks masks;
    cv::inRange(hls, cv::Scalar(0, params.whiteLightnessMin, 0),
                cv::Scalar(180, 255, 255), masks.white);
    cv::inRange(hls, params.yellowLow, params.yellowHigh, masks.yellow);

    if (params.useGradient) {
        std::vector<cv::Mat> channels;
        cv::split(hls, channels);  // gradient on lightness, not raw colour

        cv::Mat sobelX;
        cv::Sobel(channels[1], sobelX, CV_32F, 1, 0, params.sobelKernel);
        cv::Mat absX;
        cv::convertScaleAbs(sobelX, absX);
        cv::inRange(absX, cv::Scalar(params.gradientMin),
                    cv::Scalar(params.gradientMax), masks.gradient);
    } else {
        masks.gradient = cv::Mat::zeros(bgr.size(), CV_8UC1);
    }

    cv::bitwise_or(masks.white, masks.yellow, masks.combined);
    cv::bitwise_or(masks.combined, masks.gradient, masks.combined);
    return masks;
}

cv::Mat buildBinary(const cv::Mat& bgr, const ThresholdParams& params) {
    return buildBinaryMasks(bgr, params).combined;
}

} // namespace lanedet
