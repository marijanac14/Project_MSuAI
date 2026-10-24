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

} // namespace

AdvancedLaneDetector::AdvancedLaneDetector(const AdvancedParams& params)
    : params_(params) {}

cv::Mat AdvancedLaneDetector::process(const cv::Mat& frameBGR) {
    const cv::Mat binary = buildBinary(frameBGR, params_.threshold);
    const Perspective perspective(frameBGR.size(), params_.perspective);
    const cv::Mat warped = perspective.warp(binary);
    const LaneFitResult fit = fitLanes(warped, params_.sliding);

    cv::Mat output = frameBGR.clone();
    if (!fit.left.valid || !fit.right.valid) {
        return output;  // no reliable lane this frame
    }

    const int h = frameBGR.rows;
    const int w = frameBGR.cols;

    // Draw the lane in the bird's-eye view, then warp it back onto the road.
    const std::vector<cv::Point> leftCurve = curvePoints(fit.left, h);
    const std::vector<cv::Point> rightCurve = curvePoints(fit.right, h);

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
        0.5 * (curvatureMeters(fit.left, h - 1, params_.xMetersPerPixel,
                               params_.yMetersPerPixel) +
               curvatureMeters(fit.right, h - 1, params_.xMetersPerPixel,
                               params_.yMetersPerPixel));

    const double laneCenter = 0.5 * (fit.left.xAt(h - 1) + fit.right.xAt(h - 1));
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
