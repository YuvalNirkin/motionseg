// Includes
#include "motionseg/motionseg.h"

// Implementations
namespace motionseg
{
MotionSeg::~MotionSeg() {}

void MotionSeg::addFrame(const cv::Mat& frame, int frame_id) {}

void MotionSeg::writeSummary(const std::string& fileName) const {}

int MotionSeg::getCurrFrameID() const { return 0; }

cv::Mat MotionSeg::getCurrSegmentation() const { return cv::Mat(); }

cv::Mat MotionSeg::drawCurrSegmentation() const { return cv::Mat(); }

}      // namespace motionseg