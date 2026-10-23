#include "lanedet/SimpleLaneDetector.hpp"

#include <cmath>

namespace lanedet {

SimpleLaneDetector::SimpleLaneDetector(const SimpleParams& params)
    : params_(params) {}

std::vector<cv::Point> SimpleLaneDetector::roiPolygon(
    const cv::Size& size) const {
    const double w = static_cast<double>(size.width);
    const double h = static_cast<double>(size.height);

    // Trapezoid: wide at the bottom of the frame, narrow near the horizon.
    return {
        cv::Point(static_cast<int>(params_.roiBottomLeftX * w),
                  static_cast<int>(params_.roiBottomY * h)),
        cv::Point(static_cast<int>(params_.roiApexLeftX * w),
                  static_cast<int>(params_.roiApexY * h)),
        cv::Point(static_cast<int>(params_.roiApexRightX * w),
                  static_cast<int>(params_.roiApexY * h)),
        cv::Point(static_cast<int>(params_.roiBottomRightX * w),
                  static_cast<int>(params_.roiBottomY * h)),
    };
}

cv::Mat SimpleLaneDetector::applyRoi(const cv::Mat& edges) const {
    // Build a blank mask, draw the trapezoidal region we want to keep on it,
    // then AND it with the edge image so that only edges on the road survive.
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> poly{roiPolygon(edges.size())};
    cv::fillPoly(mask, poly, cv::Scalar(255));

    cv::Mat masked;
    cv::bitwise_and(edges, mask, masked);
    return masked;
}

std::optional<LaneLine> SimpleLaneDetector::fitLaneLine(
    const std::vector<cv::Vec4i>& segments) const {
    // Collect both endpoints of every segment assigned to this side, then fit
    // a single straight line through all of them with least squares (fitLine).
    std::vector<cv::Point2f> points;
    points.reserve(segments.size() * 2);
    for (const cv::Vec4i& s : segments) {
        points.emplace_back(static_cast<float>(s[0]), static_cast<float>(s[1]));
        points.emplace_back(static_cast<float>(s[2]), static_cast<float>(s[3]));
    }

    // Need at least two points (one segment) to define a line.
    if (points.size() < 2) {
        return std::nullopt;
    }

    // fitLine returns a normalised direction vector (vx, vy) and a point
    // (x0, y0) on the line. We model x as a function of y, so the slope is
    // dx/dy = vx/vy. A near-zero vy would mean a horizontal line, which the
    // slope filter should already have removed.
    cv::Vec4f line;
    cv::fitLine(points, line, cv::DIST_L2, 0, 0.01, 0.01);
    const double vx = line[0];
    const double vy = line[1];
    const double x0 = line[2];
    const double y0 = line[3];

    if (std::fabs(vy) < 1e-6) {
        return std::nullopt;
    }

    LaneLine result;
    result.slope = vx / vy;                       // dx/dy
    result.intercept = x0 - result.slope * y0;    // x where y = 0
    return result;
}

void SimpleLaneDetector::drawLaneLine(cv::Mat& image,
                                      const LaneLine& line) const {
    const double yBottom = params_.roiBottomY * image.rows;
    const double yTop = params_.roiApexY * image.rows;

    cv::Point p1(static_cast<int>(std::lround(line.xAt(yBottom))),
                 static_cast<int>(std::lround(yBottom)));
    cv::Point p2(static_cast<int>(std::lround(line.xAt(yTop))),
                 static_cast<int>(std::lround(yTop)));

    cv::line(image, p1, p2, params_.laneColor, params_.lineThickness,
             cv::LINE_AA);
}

cv::Mat SimpleLaneDetector::process(const cv::Mat& frameBGR) {
    SimpleStages ignored;
    return process(frameBGR, ignored);
}

cv::Mat SimpleLaneDetector::process(const cv::Mat& frameBGR,
                                    SimpleStages& stages) {
    CV_Assert(!frameBGR.empty());

    // --- Step 1: convert to grayscale ---
    cv::cvtColor(frameBGR, stages.gray, cv::COLOR_BGR2GRAY);

    // --- Step 2: blur and detect edges with Canny ---
    cv::Mat blurred;
    cv::GaussianBlur(stages.gray, blurred,
                     cv::Size(params_.blurKernel, params_.blurKernel), 0);
    cv::Canny(blurred, stages.edges, params_.cannyLow, params_.cannyHigh, 3);

    // --- Step 3: keep only the road region (ROI mask) ---
    stages.roiEdges = applyRoi(stages.edges);

    // --- Step 4: detect line segments with the Hough transform ---
    std::vector<cv::Vec4i> segments;
    cv::HoughLinesP(stages.roiEdges, segments, params_.houghRho,
                    params_.houghThetaDeg * CV_PI / 180.0,
                    params_.houghThreshold, params_.houghMinLineLength,
                    params_.houghMaxLineGap);

    // Raw Hough segments drawn on the frame, kept for the writeup to show the
    // stage before the segments are averaged into lane lines.
    stages.houghOverlay = frameBGR.clone();
    for (const cv::Vec4i& s : segments) {
        cv::line(stages.houghOverlay, cv::Point(s[0], s[1]),
                 cv::Point(s[2], s[3]), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }

    // --- Step 5: split segments into left / right by slope ---
    // Turn the raw Hough segments into two clean lane lines. In image
    // coordinates y grows downwards, so the left marking runs bottom-left ->
    // up-right (negative slope) and the right marking the opposite (positive
    // slope). Near-horizontal segments are rejected as noise.
    std::vector<cv::Vec4i> leftSegments;
    std::vector<cv::Vec4i> rightSegments;
    const double midX = frameBGR.cols / 2.0;

    for (const cv::Vec4i& s : segments) {
        const double dx = static_cast<double>(s[2] - s[0]);
        const double dy = static_cast<double>(s[3] - s[1]);
        if (std::fabs(dx) < 1e-3) {
            continue;  // (almost) vertical: ambiguous sign, skip
        }
        const double slope = dy / dx;
        if (std::fabs(slope) < params_.minAbsSlope) {
            continue;  // near-horizontal: not a lane line
        }
        const double segMidX = (s[0] + s[2]) / 2.0;

        // Left line: negative slope and sitting on the left half of the road.
        if (slope < 0.0 && segMidX < midX * 1.1) {
            leftSegments.push_back(s);
        }
        // Right line: positive slope and on the right half.
        else if (slope > 0.0 && segMidX > midX * 0.9) {
            rightSegments.push_back(s);
        }
    }

    std::optional<LaneLine> left = fitLaneLine(leftSegments);
    std::optional<LaneLine> right = fitLaneLine(rightSegments);

    // Bridge missing detections with the previous frame's fit (useful in video,
    // harmless on a single image where there is no previous fit).
    if (left) {
        lastLeft_ = left;
    } else {
        left = lastLeft_;
    }
    if (right) {
        lastRight_ = right;
    } else {
        right = lastRight_;
    }

    // --- Step 6: draw the result -------------------------------------------
    cv::Mat output = frameBGR.clone();

    // Shade the drivable area between the two lines (green, semi-transparent).
    if (left && right) {
        const double yBottom = params_.roiBottomY * output.rows;
        const double yTop = params_.roiApexY * output.rows;
        std::vector<cv::Point> area{
            cv::Point(static_cast<int>(std::lround(left->xAt(yBottom))),
                      static_cast<int>(std::lround(yBottom))),
            cv::Point(static_cast<int>(std::lround(left->xAt(yTop))),
                      static_cast<int>(std::lround(yTop))),
            cv::Point(static_cast<int>(std::lround(right->xAt(yTop))),
                      static_cast<int>(std::lround(yTop))),
            cv::Point(static_cast<int>(std::lround(right->xAt(yBottom))),
                      static_cast<int>(std::lround(yBottom))),
        };
        cv::Mat overlay = output.clone();
        std::vector<std::vector<cv::Point>> areaPoly{area};
        cv::fillPoly(overlay, areaPoly, params_.fillColor);
        cv::addWeighted(overlay, params_.fillAlpha, output,
                        1.0 - params_.fillAlpha, 0.0, output);
    }

    if (left) {
        drawLaneLine(output, *left);
    }
    if (right) {
        drawLaneLine(output, *right);
    }

    return output;
}

} // namespace lanedet
