#include "lanedet/AdvancedLaneDetector.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace lanedet {

namespace {

// Radius of curvature in metres at image row yPx. The pixel-space coefficients
// are rescaled to metres before applying the standard formula.
double curvatureMeters(const LanePolynomial& poly, double yPx, double xm,
                       double ym) {
    const double a = poly.a * xm / (ym * ym);
    const double b = poly.b * xm / ym;
    const double y = yPx * ym;
    return std::pow(1.0 + std::pow(2.0 * a * y + b, 2.0), 1.5) /
           std::fabs(2.0 * a);
}

std::vector<cv::Point> curvePoints(const LanePolynomial& poly, int height) {
    std::vector<cv::Point> pts;
    pts.reserve(height);
    for (int y = 0; y < height; ++y) {
        pts.emplace_back(cvRound(poly.xAt(y)), y);
    }
    return pts;
}

LanePolynomial blend(const LanePolynomial& fresh, const LanePolynomial& prev,
                     double alpha) {
    LanePolynomial out;
    out.a = alpha * fresh.a + (1.0 - alpha) * prev.a;
    out.b = alpha * fresh.b + (1.0 - alpha) * prev.b;
    out.c = alpha * fresh.c + (1.0 - alpha) * prev.c;
    out.valid = true;
    return out;
}

// Reject implausible fits: lane too narrow/wide at the car, lines crossing at
// the horizon, or the width flaring unrealistically between car and horizon.
bool plausibleLane(const LanePolynomial& left, const LanePolynomial& right,
                   int width, int height, const AdvancedParams& p) {
    const double wBottom = right.xAt(height - 1) - left.xAt(height - 1);
    const double wTop = right.xAt(0) - left.xAt(0);
    if (wBottom < p.minLaneWidthFrac * width ||
        wBottom > p.maxLaneWidthFrac * width) {
        return false;
    }
    if (wTop <= 0.0) {
        return false;
    }
    const double ratio = wTop / wBottom;
    return ratio <= p.maxWidthRatio && ratio >= 1.0 / p.maxWidthRatio;
}

} // namespace

AdvancedLaneDetector::AdvancedLaneDetector(const AdvancedParams& params)
    : params_(params) {}

cv::Mat AdvancedLaneDetector::process(const cv::Mat& frameBGR) {
    const cv::Mat binary = buildBinary(frameBGR, params_.threshold);
    const Perspective perspective(frameBGR.size(), params_.perspective);
    const cv::Mat warped = perspective.warp(binary);

    // On video, steer the base search toward the previously tracked lines so a
    // strong adjacent-lane marking cannot be mistaken for this lane's boundary.
    SlidingWindowParams sliding = params_.sliding;
    if (haveState_) {
        sliding.leftBaseHint = cvRound(smoothedLeft_.xAt(warped.rows - 1));
        sliding.rightBaseHint = cvRound(smoothedRight_.xAt(warped.rows - 1));
    }
    const LaneFitResult fit = fitLanes(warped, sliding);

    cv::Mat output = frameBGR.clone();
    const int h = frameBGR.rows;
    const int w = frameBGR.cols;

    const bool rawOk = fit.left.valid && fit.right.valid;
    bool trusted = rawOk && plausibleLane(fit.left, fit.right, w, h, params_);

    // Once we have history, also require the lane width to stay consistent: a
    // real lane does not suddenly widen (the usual symptom of the fit latching
    // onto a crack or stain) or collapse from one frame to the next.
    if (trusted && haveState_) {
        const double smoothedWidth =
            smoothedRight_.xAt(h - 1) - smoothedLeft_.xAt(h - 1);
        const double rawWidth = fit.right.xAt(h - 1) - fit.left.xAt(h - 1);
        trusted = std::fabs(rawWidth - smoothedWidth) <=
                  params_.widthTolerance * smoothedWidth;
    }

    // An implausible fit is never drawn: coast on the last good one, or draw
    // nothing until a trustworthy frame appears. A per-image detector has no
    // history, so a plausible image still draws exactly as before.
    LanePolynomial left;
    LanePolynomial right;
    bool draw = false;

    if (trusted) {
        if (haveState_) {
            smoothedLeft_ =
                blend(fit.left, smoothedLeft_, params_.smoothingAlpha);
            smoothedRight_ =
                blend(fit.right, smoothedRight_, params_.smoothingAlpha);
        } else {
            smoothedLeft_ = fit.left;
            smoothedRight_ = fit.right;
            haveState_ = true;
        }
        coastCount_ = 0;
        left = smoothedLeft_;
        right = smoothedRight_;
        draw = true;
    } else if (haveState_ && coastCount_ < params_.maxCoastFrames) {
        ++coastCount_;
        left = smoothedLeft_;
        right = smoothedRight_;
        draw = true;
    } else {
        haveState_ = false;  // nothing reliable: lane lost
    }

    if (!draw) {
        return output;
    }

    const std::vector<cv::Point> leftCurve = curvePoints(left, h);
    const std::vector<cv::Point> rightCurve = curvePoints(right, h);

    std::vector<cv::Point> area = leftCurve;
    area.insert(area.end(), rightCurve.rbegin(), rightCurve.rend());

    cv::Mat laneLayer = cv::Mat::zeros(frameBGR.size(), CV_8UC3);
    cv::fillPoly(laneLayer, std::vector<std::vector<cv::Point>>{area},
                 cv::Scalar(0, 255, 0));
    cv::polylines(laneLayer, leftCurve, false, cv::Scalar(0, 0, 255), 25);
    cv::polylines(laneLayer, rightCurve, false, cv::Scalar(255, 0, 0), 25);

    cv::Mat unwarped = perspective.unwarp(laneLayer);
    cv::addWeighted(output, 1.0, unwarped, 0.4, 0.0, output);

    const double radius =
        0.5 * (curvatureMeters(left, h - 1, params_.xMetersPerPixel,
                               params_.yMetersPerPixel) +
               curvatureMeters(right, h - 1, params_.xMetersPerPixel,
                               params_.yMetersPerPixel));

    const double laneCenter = 0.5 * (left.xAt(h - 1) + right.xAt(h - 1));
    const double offset = (w / 2.0 - laneCenter) * params_.xMetersPerPixel;

    char curvatureText[128];
    char offsetText[128];
    std::snprintf(curvatureText, sizeof(curvatureText),
                  "Radius of curvature: %.0f m", radius);
    std::snprintf(offsetText, sizeof(offsetText),
                  "Vehicle is %.2f m %s of center", std::fabs(offset),
                  offset >= 0.0 ? "right" : "left");
    cv::putText(output, curvatureText, {40, 50}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(output, offsetText, {40, 95}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    return output;
}

} // namespace lanedet
