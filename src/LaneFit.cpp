#include "lanedet/LaneFit.hpp"

#include <vector>

namespace lanedet {

namespace {

LanePolynomial fitPoly2(const std::vector<cv::Point>& pts) {
    LanePolynomial poly;
    if (pts.size() < 3) {
        return poly;
    }

    const int n = static_cast<int>(pts.size());
    cv::Mat a(n, 3, CV_64F);
    cv::Mat b(n, 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double y = pts[i].y;
        a.at<double>(i, 0) = y * y;
        a.at<double>(i, 1) = y;
        a.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = pts[i].x;
    }

    cv::Mat coeffs;
    if (!cv::solve(a, b, coeffs, cv::DECOMP_QR)) {
        return poly;
    }
    poly.a = coeffs.at<double>(0);
    poly.b = coeffs.at<double>(1);
    poly.c = coeffs.at<double>(2);
    poly.valid = true;
    return poly;
}

int columnPeak(const cv::Mat& histogram, int xFrom, int xTo) {
    int bestX = xFrom;
    int bestValue = -1;
    for (int x = xFrom; x < xTo; ++x) {
        const int value = histogram.at<int>(0, x);
        if (value > bestValue) {
            bestValue = value;
            bestX = x;
        }
    }
    return bestX;
}

} // namespace

LaneFitResult fitLanes(const cv::Mat& warpedBinary,
                       const SlidingWindowParams& params) {
    cv::Mat binary;
    if (warpedBinary.channels() != 1) {
        cv::cvtColor(warpedBinary, binary, cv::COLOR_BGR2GRAY);
    } else {
        binary = warpedBinary.clone();
    }
    // warpPerspective interpolation can leave grey values behind.
    cv::threshold(binary, binary, 1, 255, cv::THRESH_BINARY);

    const int h = binary.rows;
    const int w = binary.cols;

    // The two tallest columns in the bottom half are where each line starts.
    cv::Mat bottom = binary(cv::Range(h / 2, h), cv::Range::all());
    cv::Mat histogram;
    cv::reduce(bottom, histogram, 0, cv::REDUCE_SUM, CV_32S);
    int leftX = columnPeak(histogram, 0, w / 2);
    int rightX = columnPeak(histogram, w / 2, w);

    std::vector<cv::Point> nonzero;
    cv::findNonZero(binary, nonzero);

    const int windowHeight = h / params.windowCount;
    std::vector<cv::Point> leftPts;
    std::vector<cv::Point> rightPts;

    LaneFitResult result;
    cv::cvtColor(binary, result.visualization, cv::COLOR_GRAY2BGR);

    for (int win = 0; win < params.windowCount; ++win) {
        const int yHigh = h - win * windowHeight;
        const int yLow = yHigh - windowHeight;

        const cv::Rect leftBox(leftX - params.margin, yLow, 2 * params.margin,
                               windowHeight);
        const cv::Rect rightBox(rightX - params.margin, yLow, 2 * params.margin,
                                windowHeight);
        cv::rectangle(result.visualization, leftBox, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(result.visualization, rightBox, cv::Scalar(0, 255, 0), 2);

        long leftSum = 0, leftN = 0, rightSum = 0, rightN = 0;
        for (const cv::Point& p : nonzero) {
            if (p.y < yLow || p.y >= yHigh) {
                continue;
            }
            if (p.x >= leftBox.x && p.x < leftBox.x + leftBox.width) {
                leftPts.push_back(p);
                leftSum += p.x;
                ++leftN;
            } else if (p.x >= rightBox.x && p.x < rightBox.x + rightBox.width) {
                rightPts.push_back(p);
                rightSum += p.x;
                ++rightN;
            }
        }
        if (leftN > params.minPixels) {
            leftX = static_cast<int>(leftSum / leftN);
        }
        if (rightN > params.minPixels) {
            rightX = static_cast<int>(rightSum / rightN);
        }
    }

    for (const cv::Point& p : leftPts) {
        result.visualization.at<cv::Vec3b>(p) = cv::Vec3b(0, 0, 255);
    }
    for (const cv::Point& p : rightPts) {
        result.visualization.at<cv::Vec3b>(p) = cv::Vec3b(255, 0, 0);
    }

    result.left = fitPoly2(leftPts);
    result.right = fitPoly2(rightPts);

    for (int y = 0; y < h; ++y) {
        if (result.left.valid) {
            const int x = static_cast<int>(result.left.xAt(y));
            if (x >= 0 && x < w) {
                cv::circle(result.visualization, {x, y}, 2, {0, 255, 255}, -1);
            }
        }
        if (result.right.valid) {
            const int x = static_cast<int>(result.right.xAt(y));
            if (x >= 0 && x < w) {
                cv::circle(result.visualization, {x, y}, 2, {0, 255, 255}, -1);
            }
        }
    }

    return result;
}

} // namespace lanedet
