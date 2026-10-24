#pragma once

#include "lanedet/LaneFit.hpp"
#include "lanedet/Perspective.hpp"
#include "lanedet/Threshold.hpp"

#include <opencv2/opencv.hpp>

namespace lanedet {

struct AdvancedParams {
    ThresholdParams threshold;
    PerspectiveParams perspective;
    SlidingWindowParams sliding;

    // Metres per pixel in the warped image (approximate, US lane geometry):
    // a lane is ~3.7 m wide and the warped section is ~30 m long.
    double xMetersPerPixel = 3.7 / 700.0;
    double yMetersPerPixel = 30.0 / 720.0;
};

// Full advanced pipeline: binary mask -> bird's-eye -> sliding-window fit ->
// curvature and offset -> lane drawn back onto the road.
class AdvancedLaneDetector {
public:
    explicit AdvancedLaneDetector(const AdvancedParams& params = {});

    cv::Mat process(const cv::Mat& frameBGR);

private:
    AdvancedParams params_;
};

} // namespace lanedet
