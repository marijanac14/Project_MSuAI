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

    // Temporal smoothing across video frames. A fresh detector is created per
    // still image, so single images keep no history and are unaffected.
    double smoothingAlpha = 0.2;     // weight of the new frame against history
    int maxCoastFrames = 10;         // frames to reuse the last good fit for
    double minLaneWidthFrac = 0.20;  // sanity bounds on lane width at the car
    double maxLaneWidthFrac = 0.70;
    double maxWidthRatio = 2.5;      // allowed top/bottom lane-width ratio
    double widthTolerance = 0.25;    // allowed frame-to-frame lane-width change
};

// Full advanced pipeline: binary mask -> bird's-eye -> sliding-window fit ->
// curvature and offset -> lane drawn back onto the road. On video it smooths
// the fit over time and briefly coasts on the last good fit when a frame fails.
class AdvancedLaneDetector {
public:
    explicit AdvancedLaneDetector(const AdvancedParams& params = {});

    cv::Mat process(const cv::Mat& frameBGR);

private:
    AdvancedParams params_;

    LanePolynomial smoothedLeft_;
    LanePolynomial smoothedRight_;
    bool haveState_ = false;
    int coastCount_ = 0;
};

} // namespace lanedet
