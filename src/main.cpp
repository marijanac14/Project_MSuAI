#include "lanedet/SimpleLaneDetector.hpp"
#include "lanedet/Threshold.hpp"
#include "lanedet/Perspective.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using lanedet::SimpleLaneDetector;
using lanedet::SimpleStages;

namespace {

void printUsage(const char* exe) {
    std::cout
        << "Lane detection - Phase A (Canny + ROI + HoughLinesP)\n\n"
        << "Usage:\n"
        << "  " << exe << " image <input> [output] [--show] [--stages]\n"
        << "  " << exe << " video <input> [output] [--show]\n"
        << "  " << exe << " binary <input> [output] [--stages]\n"
        << "  " << exe << " warp <input> [--binary]\n\n"
        << "Options:\n"
        << "  --show     display the result in a window (ESC to close)\n"
        << "  --stages   (image only) also save gray/edges/roi/hough images\n\n"
        << "Examples:\n"
        << "  " << exe << " image test_images/solidWhiteRight.jpg --stages\n"
        << "  " << exe << " video test_videos/project_video01.mp4\n";
}

bool hasFlag(const std::vector<std::string>& args, const std::string& flag) {
    for (const std::string& a : args) {
        if (a == flag) {
            return true;
        }
    }
    return false;
}

// First argument that is not a flag (does not start with "--"), or "".
std::string positional(const std::vector<std::string>& args, std::size_t idx) {
    std::size_t count = 0;
    for (const std::string& a : args) {
        if (a.rfind("--", 0) == 0) {
            continue;
        }
        if (count == idx) {
            return a;
        }
        ++count;
    }
    return {};
}

int runImage(const std::vector<std::string>& args) {
    const std::string input = positional(args, 1);
    if (input.empty()) {
        std::cerr << "error: missing input image path\n";
        return 2;
    }

    cv::Mat frame = cv::imread(input);
    if (frame.empty()) {
        std::cerr << "error: could not read image '" << input << "'\n";
        return 1;
    }

    // Default output path: output_images/<stem>_lanes.jpg
    std::string output = positional(args, 2);
    if (output.empty()) {
        fs::create_directories("output_images");
        output = "output_images/" + fs::path(input).stem().string() +
                 "_lanes.jpg";
    }

    SimpleLaneDetector detector;
    SimpleStages stages;
    cv::Mat result = detector.process(frame, stages);

    if (!cv::imwrite(output, result)) {
        std::cerr << "error: could not write '" << output << "'\n";
        return 1;
    }
    std::cout << "wrote " << output << "\n";

    if (hasFlag(args, "--stages")) {
        const std::string stem = fs::path(input).stem().string();
        fs::create_directories("output_images");
        const std::string dir = "output_images/";
        cv::imwrite(dir + stem + "_1_gray.jpg", stages.gray);
        cv::imwrite(dir + stem + "_2_edges.jpg", stages.edges);
        cv::imwrite(dir + stem + "_3_roi_edges.jpg", stages.roiEdges);
        cv::imwrite(dir + stem + "_4_hough.jpg", stages.houghOverlay);
        std::cout << "wrote intermediate stages to " << dir << "\n";
    }

    if (hasFlag(args, "--show")) {
        cv::imshow("Lane detection", result);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}

int runVideo(const std::vector<std::string>& args) {
    const std::string input = positional(args, 1);
    if (input.empty()) {
        std::cerr << "error: missing input video path\n";
        return 2;
    }

    cv::VideoCapture cap(input);
    if (!cap.isOpened()) {
        std::cerr << "error: could not open video '" << input << "'\n";
        return 1;
    }

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
        fps = 25.0;  // fall back to a sane default if the container lacks it
    }

    std::string output = positional(args, 2);
    if (output.empty()) {
        fs::create_directories("output_videos");
        output = "output_videos/" + fs::path(input).stem().string() +
                 "_lanes.mp4";
    }

    cv::VideoWriter writer(output,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "error: could not open writer for '" << output << "'\n";
        return 1;
    }

    const bool show = hasFlag(args, "--show");
    SimpleLaneDetector detector;
    cv::Mat frame;
    long long frameCount = 0;

    // Main loop: read, process and write one frame at a time.
    while (cap.read(frame)) {
        cv::Mat result = detector.process(frame);
        writer.write(result);
        ++frameCount;

        if (show) {
            cv::imshow("Lane detection", result);
            if ((cv::waitKey(1) & 0xFF) == 27) {  // ESC
                break;
            }
        }
    }

    std::cout << "processed " << frameCount << " frames, wrote " << output
              << "\n";
    if (show) {
        cv::destroyAllWindows();
    }
    return 0;
}

// Phase B, step 1: write out the binary lane mask so we can inspect and tune it.
int runBinary(const std::vector<std::string>& args) {
    const std::string input = positional(args, 1);
    if (input.empty()) {
        std::cerr << "error: missing input image path\n";
        return 2;
    }

    cv::Mat frame = cv::imread(input);
    if (frame.empty()) {
        std::cerr << "error: could not read image '" << input << "'\n";
        return 1;
    }

    std::string output = positional(args, 2);
    if (output.empty()) {
        fs::create_directories("output_images");
        output = "output_images/" + fs::path(input).stem().string() +
                 "_binary.jpg";
    }

    lanedet::ThresholdMasks masks = lanedet::buildBinaryMasks(frame);
    if (!cv::imwrite(output, masks.combined)) {
        std::cerr << "error: could not write '" << output << "'\n";
        return 1;
    }
    std::cout << "wrote " << output << "\n";

    if (hasFlag(args, "--stages")) {
        const std::string stem = fs::path(input).stem().string();
        const std::string dir = "output_images/";
        cv::imwrite(dir + stem + "_b_white.jpg", masks.white);
        cv::imwrite(dir + stem + "_b_yellow.jpg", masks.yellow);
        cv::imwrite(dir + stem + "_b_gradient.jpg", masks.gradient);
        std::cout << "wrote binary sub-masks to " << dir << "\n";
    }
    return 0;
}

// Phase B, step 2: show the bird's-eye warp and the source trapezoid, so we can
// check that straight lanes come out vertical and parallel.
int runWarp(const std::vector<std::string>& args) {
    const std::string input = positional(args, 1);
    if (input.empty()) {
        std::cerr << "error: missing input image path\n";
        return 2;
    }

    cv::Mat frame = cv::imread(input);
    if (frame.empty()) {
        std::cerr << "error: could not read image '" << input << "'\n";
        return 1;
    }

    fs::create_directories("output_images");
    const std::string dir = "output_images/";
    const std::string stem = fs::path(input).stem().string();

    lanedet::Perspective perspective(frame.size());

    cv::Mat overlay = frame.clone();
    std::vector<cv::Point> trapezoid;
    for (const cv::Point2f& p : perspective.source()) {
        trapezoid.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::polylines(overlay, trapezoid, true, cv::Scalar(0, 0, 255), 2,
                  cv::LINE_AA);
    cv::imwrite(dir + stem + "_src.jpg", overlay);

    cv::imwrite(dir + stem + "_warp.jpg", perspective.warp(frame));

    if (hasFlag(args, "--binary")) {
        cv::Mat warpedBinary = perspective.warp(lanedet::buildBinary(frame));
        cv::imwrite(dir + stem + "_warp_binary.jpg", warpedBinary);
    }

    std::cout << "wrote warp output to " << dir << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 2;
    }

    const std::string mode = argv[1];
    const std::vector<std::string> args(argv + 1, argv + argc);

    if (mode == "image") {
        return runImage(args);
    }
    if (mode == "video") {
        return runVideo(args);
    }
    if (mode == "binary") {
        return runBinary(args);
    }
    if (mode == "warp") {
        return runWarp(args);
    }

    std::cerr << "error: unknown mode '" << mode << "'\n\n";
    printUsage(argv[0]);
    return 2;
}
