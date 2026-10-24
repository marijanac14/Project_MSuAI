#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace lanedet {

// Corners of the bird's-eye warp, given as fractions of the frame so they hold
// at any resolution. The source is a trapezoid lying on the road (narrow near
// the horizon, wide at the car); it maps to a rectangle inset by dstMargin.
// Starting values are tuned on the straight-line highway images.
struct PerspectiveParams {
    cv::Point2f srcTopLeft{0.457f, 0.64f};
    cv::Point2f srcTopRight{0.549f, 0.64f};
    cv::Point2f srcBottomRight{0.88f, 0.96f};
    cv::Point2f srcBottomLeft{0.16f, 0.96f};
    float dstMargin = 0.25f;
};

class Perspective {
public:
    Perspective(const cv::Size& frameSize,
                const PerspectiveParams& params = {});

    cv::Mat warp(const cv::Mat& img) const;    // front view -> bird's-eye
    cv::Mat unwarp(const cv::Mat& img) const;  // bird's-eye -> front view

    const std::vector<cv::Point2f>& source() const { return src_; }
    const std::vector<cv::Point2f>& destination() const { return dst_; }

private:
    cv::Size frameSize_;
    std::vector<cv::Point2f> src_;
    std::vector<cv::Point2f> dst_;
    cv::Mat forward_;
    cv::Mat inverse_;
};

} // namespace lanedet
