#include "lanedet/Perspective.hpp"

namespace lanedet {

Perspective::Perspective(const cv::Size& frameSize,
                         const PerspectiveParams& params)
    : frameSize_(frameSize) {
    const float w = static_cast<float>(frameSize.width);
    const float h = static_cast<float>(frameSize.height);

    src_ = {
        {params.srcTopLeft.x * w, params.srcTopLeft.y * h},
        {params.srcTopRight.x * w, params.srcTopRight.y * h},
        {params.srcBottomRight.x * w, params.srcBottomRight.y * h},
        {params.srcBottomLeft.x * w, params.srcBottomLeft.y * h},
    };

    const float m = params.dstMargin * w;
    dst_ = {
        {m, 0.0f},
        {w - m, 0.0f},
        {w - m, h},
        {m, h},
    };

    forward_ = cv::getPerspectiveTransform(src_, dst_);
    inverse_ = cv::getPerspectiveTransform(dst_, src_);
}

cv::Mat Perspective::warp(const cv::Mat& img) const {
    cv::Mat out;
    cv::warpPerspective(img, out, forward_, frameSize_, cv::INTER_LINEAR);
    return out;
}

cv::Mat Perspective::unwarp(const cv::Mat& img) const {
    cv::Mat out;
    cv::warpPerspective(img, out, inverse_, frameSize_, cv::INTER_LINEAR);
    return out;
}

} // namespace lanedet
