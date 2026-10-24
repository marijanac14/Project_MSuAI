#pragma once

#include <opencv2/opencv.hpp>

namespace lanedet {

// A lane boundary as a second-order curve x = a*y^2 + b*y + c. x is expressed
// as a function of y because, after the bird's-eye warp, the lines run almost
// vertically.
struct LanePolynomial {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    bool valid = false;

    double xAt(double y) const { return (a * y + b) * y + c; }
};

struct SlidingWindowParams {
    int windowCount = 9;
    int margin = 100;    // half width of each search window, in pixels
    int minPixels = 50;  // pixels needed before a window recenters
};

struct LaneFitResult {
    LanePolynomial left;
    LanePolynomial right;
    cv::Mat visualization;  // warped mask in colour, with windows and fits
};

// Find lane pixels with a sliding window over the warped binary image and fit
// a polynomial to each side.
LaneFitResult fitLanes(const cv::Mat& warpedBinary,
                       const SlidingWindowParams& params = {});

} // namespace lanedet
