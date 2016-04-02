#ifndef __FACE_MOTIONSEG_HPP__
#define __FACE_MOTIONSEG_HPP__

// Includes
#include "motionseg/motionseg.h"
#include <string>
#include <memory>
#include <opencv2/core.hpp>

// Namespaces

// Definitions
namespace motionseg
{
    class FaceMotionSeg : public MotionSeg
    {
    public:

        virtual ~FaceMotionSeg();

        static std::shared_ptr<FaceMotionSeg> create(const cv::Size& frame_size,
            int frame_radius, const std::string& landmarksModelPath,
            const std::string& outputPath, unsigned int verbose);
    };
}   // namespace motionseg


#endif  // __FACE_MOTIONSEG_HPP__