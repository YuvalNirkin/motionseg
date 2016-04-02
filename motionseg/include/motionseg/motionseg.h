#ifndef __MOTIONSEG_HPP__
#define __MOTIONSEG_HPP__

// Includes
#include <string>
#include <opencv2/core.hpp>

// Definitions
namespace motionseg
{
    /// Provide motion segmentation functionality
    class MotionSeg
    {
    public:

        /// Virtual destructor
        virtual ~MotionSeg();

        /** Add another frame to process the motion from.
            \param frame A frame from a video sequence
            \param frame_id Optional frame id (if not provided an internal counter
            will be used instead)
        */
        virtual void addFrame(const cv::Mat& frame, int frame_id = -1);

        /** Write a summary of the processed sequence containing the frame's
            ids and scores.
            The summary will be written in CSV (Comma-separated values) format.
            \param fileName The summary file name. If a relative path is specified 
            then it will be written in the output directory.
        */
        virtual void writeSummary(const std::string& fileName = "summary.csv") const;

        /// Get the current frame ID.
        virtual int getCurrFrameID() const;

        /** Get the current frame's segmentation.
            \return Grayscale image. 0 for background and all other values are 
            the segmentation
        */
        virtual cv::Mat getCurrSegmentation() const;

        /** Draw the current segmentation augmented with the current frame 
        */
        virtual cv::Mat drawCurrSegmentation() const;
    };
}   // namespace motionseg

#endif  // __MOTIONSEG_HPP__