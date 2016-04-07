// Includes
#include "motionseg/face_motionseg.h"

// std
#include <iostream>

// Boost
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

// vsal
#include <vsal/VideoStreamFactory.h>
#include <vsal/VideoStreamOpenCV.h>

// Namespaces
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::runtime_error;
using namespace boost::program_options;

// Implementation

int main(int argc, char* argv[])
{
    // Parse command line arguments
    string inputPath, outputPath, landmarksModelPath;
    int device;
    unsigned int width, height, verbose;
    double fps, frame_scale;
    bool preview;
    try {
        options_description desc("Allowed options");
        desc.add_options()
            ("help", "display the help message")
            ("input,i", value<string>(&inputPath), "input path")
            ("output,o", value<string>(&outputPath), "output path")
            ("device,d", value<int>(&device)->default_value(-1), "device id")
            ("width,w", value<unsigned int>(&width)->default_value(0), "frame width")
            ("height,h", value<unsigned int>(&height)->default_value(0), "frame height")
            ("fps,f", value<double>(&fps)->default_value(30.0), "frames per second")
            ("scale,s", value<double>(&frame_scale)->default_value(1.0), "frame scale")
            ("landmarks,l", value<string>(&landmarksModelPath), "path to landmarks model file")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information")
            ("preview,p", value<bool>(&preview)->default_value(false), "toggle preview loop")
            ;
        variables_map vm;
        store(command_line_parser(argc, argv).options(desc).
            positional(positional_options_description().add("input", -1)).run(), vm);
        if (vm.count("help")) {
            cout << "Usage: preview [options]" << endl;
            cout << desc << endl;
            exit(0);
        }
        notify(vm);
    }
    catch (const error& e) {
        cout << "Error while parsing command-line arguments: " << e.what() << endl;
        cout << "Use --help to display a list of options." << endl;
        exit(1);
    }

    try
    {
        bool live = device >= 0;
        if (!outputPath.empty())
            boost::filesystem::create_directory(outputPath);

        // Create video source
        vsal::VideoStreamFactory& vsf = vsal::VideoStreamFactory::getInstance();
        vsal::VideoStreamOpenCV* vs = nullptr;
        if (live) vs = (vsal::VideoStreamOpenCV*)vsf.create(device, width, height);
        else if (!inputPath.empty()) vs = (vsal::VideoStreamOpenCV*)vsf.create(inputPath);
        else throw runtime_error("No video source specified!");

        // Open video source
        vs->open();
        if (width <= 0 || height <= 0)
        {
            width = vs->getWidth();
            height = vs->getHeight();
        }

        // Initialize motion segmentation
        std::shared_ptr<motionseg::FaceMotionSeg> fmg = motionseg::FaceMotionSeg::create(
            cv::Size(width, height), 2, landmarksModelPath, outputPath, verbose);

        // Preview loop
        cv::Mat frame;
        int frameCounter = 0;
        while (preview && vs->read())
        {
            if (!vs->isUpdated()) continue;

            frame = vs->getFrame();

            // Show overlay
            cv::putText(frame, (boost::format("Frame count: %d") % ++frameCounter).str(), cv::Point(15, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255),
                1, CV_AA);
            cv::putText(frame, "press any key to start processing", cv::Point(10, frame.rows - 20),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 165, 255),
                1, CV_AA);

            // Show frame
            cv::imshow("frame", frame);
            int key = cv::waitKey(1);
            if (key >= 0) break;
        }
        cv::destroyWindow("frame");

        // Main processing loop
        cv::Mat seg_out;
        while (vs->read())
        {
            if (!vs->isUpdated()) continue;

            frame = vs->getFrame();
            fmg->addFrame(frame, ++frameCounter);
            seg_out = fmg->drawCurrSegmentation();
            if (seg_out.empty()) continue;

            // Show overlay
            cv::putText(seg_out, (boost::format("Frame: %d") % fmg->getCurrFrameID()).str(), cv::Point(15, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255),
                1, CV_AA);
            cv::putText(seg_out, "press any key to stop processing", cv::Point(10, frame.rows - 20),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 165, 255),
                1, CV_AA);

            // Show frame
            cv::imshow("frame", seg_out);
            int key = cv::waitKey(1);
            if (key >= 0) break;
        }
        fmg->writeSummary();
    }
    catch (std::exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}