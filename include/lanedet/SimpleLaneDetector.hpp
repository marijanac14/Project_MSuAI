#pragma once

#include <opencv2/opencv.hpp>

#include <optional>
#include <vector>

namespace lanedet {

/// Tunable parameters for the simple Canny + ROI + HoughLinesP pipeline.
///
/// The defaults are a sensible starting point for highway footage. The Hough
/// settings are deliberately relaxed (short minimum line length, large gap) so
/// that dashed lane markings are detected as well as solid ones.
struct SimpleParams {
    // --- Pre-processing ---
    int blurKernel = 5;              ///< Gaussian blur kernel size (odd number).

    // --- Canny edge detector ---
    double cannyLow = 100.0;
    double cannyHigh = 170.0;

    // --- Region of interest, expressed as fractions of the frame size ---
    // A trapezoid that keeps only the road area in front of the car.
    // Fractions (not pixels) so the same values work for 960x540 and 1280x720.
    double roiBottomLeftX = 0.08;    ///< bottom-left corner, x (fraction of width)
    double roiBottomRightX = 0.95;   ///< bottom-right corner, x
    double roiApexLeftX = 0.45;      ///< top-left corner, x
    double roiApexRightX = 0.55;     ///< top-right corner, x
    double roiApexY = 0.60;          ///< top edge of trapezoid (fraction of height)
    double roiBottomY = 1.00;        ///< bottom edge of trapezoid

    // --- Probabilistic Hough transform ---
    // Dashed markings are short segments, so minLineLength is kept small and
    // maxLineGap large enough to bridge the gaps between the dashes.
    double houghRho = 1.0;           ///< distance resolution in pixels
    double houghThetaDeg = 1.0;      ///< angle resolution in degrees (CV_PI/180)
    int houghThreshold = 30;
    double houghMinLineLength = 20.0;
    double houghMaxLineGap = 120.0;

    // --- Left / right classification ---
    double minAbsSlope = 0.5;        ///< reject near-horizontal segments (noise)

    // --- Drawing ---
    cv::Scalar laneColor = cv::Scalar(0, 0, 255);   ///< lane lines, red (BGR)
    cv::Scalar fillColor = cv::Scalar(0, 255, 0);   ///< lane area, green (BGR)
    int lineThickness = 8;
    double fillAlpha = 0.3;          ///< transparency of the green lane area
};

/// One lane boundary modelled as x = slope * y + intercept.
///
/// We express x as a function of y (not the usual y = f(x)) because lane lines
/// are close to vertical, so their slope in the usual sense would be huge.
struct LaneLine {
    double slope = 0.0;        ///< dx/dy
    double intercept = 0.0;    ///< x value where y = 0

    double xAt(double y) const { return slope * y + intercept; }
};

/// Intermediate pipeline images, kept so the writeup can illustrate each stage.
struct SimpleStages {
    cv::Mat gray;           ///< grayscale frame
    cv::Mat edges;          ///< Canny edges of the whole frame
    cv::Mat roiEdges;       ///< Canny edges after the ROI mask
    cv::Mat houghOverlay;   ///< raw Hough segments drawn on the frame
};

/// Phase A lane detector: grayscale -> blur -> Canny -> ROI -> HoughLinesP ->
/// split into left/right by slope -> fit one line per side -> draw.
class SimpleLaneDetector {
public:
    explicit SimpleLaneDetector(const SimpleParams& params = SimpleParams{});

    /// Run the pipeline on one BGR frame and return the annotated frame.
    cv::Mat process(const cv::Mat& frameBGR);

    /// Same as process() but also fills `stages` with the intermediate images.
    cv::Mat process(const cv::Mat& frameBGR, SimpleStages& stages);

    const SimpleParams& params() const { return params_; }

private:
    /// Build the trapezoidal ROI polygon for a frame of the given size.
    std::vector<cv::Point> roiPolygon(const cv::Size& size) const;

    /// Keep only the edges inside the ROI trapezoid.
    cv::Mat applyRoi(const cv::Mat& edges) const;

    /// Fit a single lane line through a set of Hough segments (least squares).
    /// Returns std::nullopt when there are too few points to fit.
    std::optional<LaneLine> fitLaneLine(
        const std::vector<cv::Vec4i>& segments) const;

    /// Draw one lane line between y = roiBottomY and y = roiApexY.
    void drawLaneLine(cv::Mat& image, const LaneLine& line) const;

    SimpleParams params_;

    // Last good fit per side. In video this bridges frames where one side is
    // momentarily missing (e.g. a gap between dashes), avoiding flicker.
    std::optional<LaneLine> lastLeft_;
    std::optional<LaneLine> lastRight_;
};

} // namespace lanedet
