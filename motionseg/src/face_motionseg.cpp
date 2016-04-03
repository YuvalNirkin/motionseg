// Includes
#include "motionseg/face_motionseg.h"
#include "thread_pool.h"
#include <fstream>

// Boost
#include <boost/format.hpp>//
#include <boost/filesystem.hpp>//

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudastereo.hpp>

// dlib
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>

// Implementation

namespace motionseg
{

class FaceMotionSegImpl : public FaceMotionSeg
{
public:
    FaceMotionSegImpl(const cv::Size& frame_size, int frame_radius,
        const std::string& landmarksModelPath, const std::string& outputPath,
        unsigned int verbose = 0);

    ~FaceMotionSegImpl();

    void addFrame(const cv::Mat& frame, int frame_id = -1);

    void writeSummary(const std::string& fileName = "summary.csv") const;

    int getCurrFrameID() const;

    cv::Mat getCurrSegmentation() const;

    cv::Mat drawCurrSegmentation() const;

private:

    void calcCurrFrameMotion();
    void showFloatImage(const std::string& title, const cv::Mat& img);
    bool calcMainFaceRect(const cv::Mat& frame_color, cv::Rect& face, cv::Point2f& center,
        dlib::full_object_detection& landmarks);
    dlib::rectangle& selectMainFace(const cv::Mat& img, std::vector<dlib::rectangle>& faces);
    cv::Rect fitFaceRect(const dlib::full_object_detection& shape, const cv::Size& frameSize,
        cv::Point2f& center, const cv::Size& maxRectSize = cv::Size(0, 0), bool square = false);
    float calcDynamicThreshold(const cv::Mat& W, const cv::Rect rect);
    cv::Mat calcSegmentation(const cv::Mat& frame_color, const cv::Mat& W,
        const cv::Rect rect, const cv::Point2f& center, float t);
    cv::Mat calcSegmentation_2(const cv::Mat& frame_color, const cv::Mat& W,
        const cv::Rect rect, const cv::Point2f& center, float t);
    void removeNeck(cv::Mat& seg, const cv::Rect& rect,
        const dlib::full_object_detection& landmarks);
    void removeSecondaryObjects(cv::Mat& seg, const cv::Rect& rect,
        const std::vector<std::vector<cv::Point>>& contours);
    float calcScore(const cv::Mat& frame_color, const cv::Mat& W, const cv::Mat& seg,
        const cv::Rect rect);

    void renderLandmarks(const cv::Mat& img, const dlib::full_object_detection& landmarks,
        const cv::Scalar& color = cv::Scalar(0, 255, 0));
    cv::Mat renderWeightMap(const cv::Mat& weightMap);
    void drawSegmentation(cv::Mat& frame_color, const cv::Rect& rect, const cv::Mat& seg);

    void writeSegmentation(cv::Mat& frame_color, const cv::Rect& rect, const cv::Mat& seg,
        int frame_id);

private:
    int r, n;
    int frame_counter, cfi;
    cv::Size frame_size;
    std::vector<cv::Mat> frames_color, frames_gray;
    std::vector<cv::cuda::GpuMat> d_frames_color, d_frames_gray;
    std::vector<int> frame_ids;
    cv::Mat disp_dbf;
    cv::cuda::GpuMat d_disp, d_disp_dbf;

    cv::Ptr<cv::cuda::StereoBM> bm;
    cv::Ptr<cv::cuda::DisparityBilateralFilter> dbf;

    cv::Mat seg, seg_out;

    dlib::frontal_face_detector detector;
    dlib::shape_predictor pose_model;

    std::vector<cv::Rect> rects;
    std::vector<cv::Point2f> centers;
    std::vector<dlib::full_object_detection> shapes;
    //std::queue<std::shared_ptr<std::thread>> threads;

    // threads
    ThreadPool pool;
    std::vector<std::mutex> mutexes;
    std::vector<std::future<bool>> rectsStatus;

    std::string outputPath;
    unsigned int verbose;
    std::list<std::pair<int, float>> summary;
};

FaceMotionSeg::~FaceMotionSeg() {}

std::shared_ptr<FaceMotionSeg> FaceMotionSeg::create(const cv::Size& frame_size,
    int frame_radius, const std::string& landmarksModelPath,
    const std::string& outputPath, unsigned int verbose)
{
    return std::make_shared<FaceMotionSegImpl>(frame_size, frame_radius,
        landmarksModelPath, outputPath, verbose);
}

FaceMotionSegImpl::FaceMotionSegImpl(const cv::Size& frame_size, int frame_radius,
    const std::string& landmarksModelPath, const std::string& outputPath,
    unsigned int verbose) :
    r(frame_radius), n(2 * r + 1),
    frame_counter(0),
    cfi(-r - 1),
    frame_size(frame_size),
    frames_color(n),
    frames_gray(n),
    d_frames_color(n),
    d_frames_gray(n),
    frame_ids(n),
    pool(r + 1),
    rects(n),
    centers(n),
    shapes(n),
    rectsStatus(n),
    mutexes(n),
    outputPath(outputPath),
    verbose(verbose)
{
    for (int i = 0; i < n; ++i)
    {
        frames_gray[i].create(frame_size, CV_8U);
    }

    bm = cv::cuda::createStereoBM(256, 11); // 19
    dbf = cv::cuda::createDisparityBilateralFilter(256, 11, 10);    // 19

    // Face detector for finding bounding boxes for each face in an image
    detector = dlib::get_frontal_face_detector();

    // Shape predictor for finding landmark positions given an image and face bounding box.
    dlib::deserialize(landmarksModelPath) >> pose_model;

    // Create output directory
    if (!outputPath.empty())
        boost::filesystem::create_directory(outputPath);
}

FaceMotionSegImpl::~FaceMotionSegImpl()
{
}

void FaceMotionSegImpl::addFrame(const cv::Mat& frame, int frame_id)
{
    ++frame_counter;
    cfi = (cfi + 1) % n;
    int i = (cfi + r) % n;

    {
        //std::lock_guard<std::mutex> lock(mutexes[i]);
        frame.copyTo(frames_color[i]);
        frame_ids[i] = frame_id >= 0 ? frame_id : frame_counter;
    }

    //threads.push(std::make_shared<std::thread>(&FaceMotionSegImpl::calcMainFaceRect_task, this, i));
    ///
    if (frame_counter > r)
    {
        rectsStatus[i] = (pool.enqueue([this, i]{
            return calcMainFaceRect(frames_color[i], rects[i], centers[i], shapes[i]);
        }));
    }
    ///
    cv::cvtColor(frame, frames_gray[i], cv::COLOR_BGR2GRAY);

    d_frames_color[i].upload(frame);
    d_frames_gray[i].upload(frames_gray[i]);

    if (frame_counter >= n)
        calcCurrFrameMotion();
}

void FaceMotionSegImpl::writeSummary(const std::string& fileName) const
{
    if (outputPath.empty()) return;

    boost::filesystem::path filePath = fileName;
    if (filePath.is_relative()) 
        filePath = boost::filesystem::path(outputPath) / fileName;
    std::ofstream file(filePath.string());
    file << "id, score" << std::endl;
    for (auto&& i : summary)
        file << i.first << ", " << i.second << std::endl;
}

int FaceMotionSegImpl::getCurrFrameID() const
{
    return frame_ids[cfi];
}

cv::Mat FaceMotionSegImpl::getCurrSegmentation() const
{
    return seg;
}

cv::Mat FaceMotionSegImpl::drawCurrSegmentation() const
{
    return seg_out;
}

void FaceMotionSegImpl::calcCurrFrameMotion()
{
    cv::Mat flow_src, flow_dst;
    cv::Mat W = cv::Mat::zeros(frame_size, CV_32F);
    for (int i = 0, f = (cfi - r + n) % n; i < n; ++i, f = (f + 1) % n)
    {
        if (f == cfi) continue;
        bm->compute(d_frames_gray[cfi], d_frames_gray[f], d_disp);
        dbf->apply(d_disp, d_frames_color[cfi], d_disp_dbf);
        d_disp_dbf.download(disp_dbf);

        /// 
        cv::Mat temp;
        disp_dbf.convertTo(temp, CV_32F);
        W += temp;

        //showFloatImage((boost::format("disp_dbf (%d)") % i).str(), disp_dbf);
        //cv::waitKey(1);
    }

    // Initialize segmentation drawing
    seg_out = frames_color[cfi].clone();

    // Find main face bounding box
    //cv::Rect face;
    //if (!calcMainFaceRect(frames_color[cfi], face)) return;
    //std::cout << "main " << frame_ids[cfi] << " waiting" << std::endl;
    if (!rectsStatus[cfi].get()) return;
    cv::Rect face = rects[cfi];
    cv::Point2f& center = centers[cfi];

    float t = calcDynamicThreshold(W, face);
    cv::Mat seg_thresh = calcSegmentation(frames_color[cfi], W, face, center, t);
    seg = calcSegmentation_2(frames_color[cfi], W, face, center, t);
    removeNeck(seg, face, shapes[cfi]);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(seg.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    removeSecondaryObjects(seg, face, contours);
    float score = calcScore(frames_color[cfi], W, seg, face);

    // Update summary
    summary.push_back(std::make_pair(frame_ids[cfi], score));

    cv::Mat W_out, seg_thresh_out;
    if (verbose >= 2)
    {
        W_out = renderWeightMap(W);
        cv::putText(W_out, (boost::format("Frame: %d") % frame_ids[cfi]).str(), cv::Point(15, 15),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
        cv::putText(W_out, (boost::format("Threshold: %f") % t).str(), cv::Point(15, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
        cv::putText(W_out, (boost::format("Score: %f") % score).str(), cv::Point(15, 55),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);

        seg_thresh_out = frames_color[cfi].clone();
        drawSegmentation(seg_thresh_out, face, seg_thresh);
        cv::rectangle(seg_thresh_out, face, cv::Scalar(0, 255, 0), 1);
        cv::putText(seg_thresh_out, (boost::format("Frame: %d") % frame_ids[cfi]).str(), cv::Point(15, 15),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
        cv::putText(seg_thresh_out, (boost::format("Threshold: %f") % t).str(), cv::Point(15, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
        cv::putText(seg_thresh_out, (boost::format("Score: %f") % score).str(), cv::Point(15, 55),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);

        cv::imshow("W", W_out);
        cv::imshow("seg_thresh", seg_thresh_out);
    }

    drawSegmentation(seg_out, face, seg);
    //renderLandmarks(seg_out, shapes[cfi]);
    cv::rectangle(seg_out, face, cv::Scalar(0, 255, 0), 1);

    /* 
    cv::putText(seg_out, (boost::format("Frame: %d") % frame_ids[cfi]).str(), cv::Point(15, 15),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
    cv::putText(seg_out, (boost::format("Threshold: %f") % t).str(), cv::Point(15, 35),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
    cv::putText(seg_out, (boost::format("Score: %f") % score).str(), cv::Point(15, 55),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);

    cv::imshow("seg", seg_out);
    cv::waitKey(1);
    */
    if (!outputPath.empty() && verbose)
    {
        if (verbose >= 1)
        {
            cv::imwrite(str(boost::format("%s\\seg_debug_%04d.jpg") %
                outputPath % frame_ids[cfi]), seg_out);
            if (verbose >= 2)
            {
                cv::imwrite(str(boost::format("%s\\W_%04d.jpg") %
                    outputPath % frame_ids[cfi]), W_out);
                cv::imwrite(str(boost::format("%s\\seg_thresh_%04d.jpg") %
                    outputPath % frame_ids[cfi]), seg_thresh_out);
            }
        }
    }
    /////////////
    writeSegmentation(frames_color[cfi], face, seg, frame_ids[cfi]);
    /*
    if (!outputPath.empty())
    {
        cv::Mat frame_cropped = frames_color[cfi](face);
        cv::Mat seg_cropped = seg(face);
        cv::MatSize size = frame_cropped.size;
        int max_index = std::distance(size.p, std::max_element(size.p, size.p + 2));
        if (size[max_index] > 500)
        {
            float scale = 500.0f / (float)size[max_index];
            int w = (int)std::round(frame_cropped.cols * scale);
            int h = (int)std::round(frame_cropped.rows * scale);
            cv::resize(frame_cropped, frame_cropped, cv::Size(w, h));
            cv::resize(seg_cropped, seg_cropped, cv::Size(w, h));
        }

        //cv::resize(frame_cropped, frame_cropped, cv::Size(500, 500));
        //cv::resize(seg_cropped, seg_cropped, cv::Size(500, 500));

        cv::imwrite(str(boost::format("%s\\frame_%04d.png") %
            outputPath % frame_ids[cfi]), frame_cropped);
        cv::imwrite(str(boost::format("%s\\seg_%04d.png") %
            outputPath % frame_ids[cfi]), seg_cropped);
    }
    */
}

void FaceMotionSegImpl::showFloatImage(const std::string& title, const cv::Mat& img)
{
    cv::Mat img_norm, img_color;
    cv::normalize(img, img_norm, 0, 1, cv::NORM_MINMAX, CV_32F);
    img_norm.convertTo(img_color, CV_8U, 255.0);
    cv::applyColorMap(img_color, img_color, cv::COLORMAP_JET);
    cv::imshow(title, img_color);
}

bool FaceMotionSegImpl::calcMainFaceRect(const cv::Mat& frame_color, cv::Rect& face,
    cv::Point2f& center, dlib::full_object_detection& landmarks)
{
    // Convert OpenCV's mat to dlib format
    dlib::cv_image<dlib::bgr_pixel> dlib_frame(frame_color);

    // Detect bounding boxes around all the faces in the image.
    std::vector<dlib::rectangle> faces = detector(dlib_frame);
    if (faces.empty()) return false;
    dlib::rectangle& main_face = selectMainFace(frame_color, faces);

    // Detect landmarks
    landmarks = pose_model(dlib_frame, main_face);

    // Return fitted face bounding box
    face = fitFaceRect(landmarks, frame_color.size(), center, cv::Size(0, 0), true);

    return true;
}

dlib::rectangle& FaceMotionSegImpl::selectMainFace(const cv::Mat& img,
    std::vector<dlib::rectangle>& faces)
{
    std::vector<double> scores(faces.size());
    dlib::point bl, tr;
    cv::Point face_center, img_center(img.cols / 2, img.rows / 2);
    double dist, size;
    for (size_t i = 0; i < faces.size(); ++i)
    {
        dlib::rectangle& face = faces[i];
        bl = face.bl_corner(); tr = face.tr_corner();
        face_center = cv::Point((bl.x() + tr.x()) / 2, (bl.y() + tr.y()) / 2);
        dist = cv::norm(cv::Mat(img_center - face_center), cv::NORM_L2SQR);
        size = (double)face.area();
        scores[i] = size - dist;
    }

    // Return the face with the largest score
    int max_index = std::distance(scores.begin(),
        std::max_element(scores.begin(), scores.end()));
    return faces[max_index];
}

cv::Rect FaceMotionSegImpl::fitFaceRect(const dlib::full_object_detection& shape,
    const cv::Size& frameSize, cv::Point2f& center,
    const cv::Size& maxRectSize, bool square)
{
    long xmin(std::numeric_limits<long>::max()), ymin(std::numeric_limits<long>::max()),
        xmax(-1), ymax(-1), sumx(0), sumy(0);
    for (unsigned long i = 0; i < shape.num_parts(); ++i)
    {
        xmin = std::min(xmin, shape.part(i).x());
        ymin = std::min(ymin, shape.part(i).y());
        xmax = std::max(xmax, shape.part(i).x());
        ymax = std::max(ymax, shape.part(i).y());
        sumx += shape.part(i).x();
        sumy += shape.part(i).y();
    }
    long width = xmax - xmin + 1;
    long height = ymax - ymin + 1;
    long centerx = (xmin + xmax) / 2;
    long centery = (ymin + ymax) / 2;
    long avgx = (long)std::round(sumx / shape.num_parts());
    long avgy = (long)std::round(sumy / shape.num_parts());
    long devx = centerx - avgx;
    long devy = centery - avgy;
    long dleft = (long)std::round(0.1*width) + abs(devx < 0 ? devx : 0);
    long dtop = (long)std::round(height*(std::max(float(width) / height, 1.0f)*2 - 1)) + abs(devy < 0 ? devy : 0);
    long dright = (long)std::round(0.1*width) + abs(devx > 0 ? devx : 0);
    long dbottom = (long)std::round(0.1*height) + abs(devy > 0 ? devy : 0);

    // Output center
    center.x = (xmin - dleft + xmax + dright)*0.5f;
    center.y = (ymin - dtop + ymax + dbottom)*0.5f;

    // Limit to frame boundaries
    xmin = std::max(0L, xmin - dleft);
    ymin = std::max(0L, ymin - dtop);
    xmax = std::min((long)frameSize.width - 1, xmax + dright);
    ymax = std::min((long)frameSize.height - 1, ymax + dbottom);

    // Limit dimensions
    long new_width = xmax - xmin + 1, new_height = ymax - ymin + 1;
    if (maxRectSize.width > 0 && new_width > (long)maxRectSize.width)
    {
        long dx = (long)ceil((new_width - maxRectSize.width) / 2.0);
        xmin += dx; xmax -= dx;
    }
    if (maxRectSize.height > 0 && new_height > (long)maxRectSize.height)
    {
        long dy = (long)ceil((new_height - maxRectSize.height) / 2.0);
        ymin += dy; ymax -= dy;
    }

    // Make square
    if (square)
    {
        long sq_width = std::max(xmax - xmin + 1, ymax - ymin + 1);
        centerx = (xmin + xmax) / 2;
        centery = (ymin + ymax) / 2;
        xmin = centerx - ((sq_width - 1) / 2);
        ymin = centery - ((sq_width - 1) / 2);
        xmax = xmin + sq_width - 1;
        ymax = ymin + sq_width - 1;

        // Limit to frame boundaries
        xmin = std::max(0L, xmin);
        ymin = std::max(0L, ymin);
        xmax = std::min((long)frameSize.width - 1, xmax);
        ymax = std::min((long)frameSize.height - 1, ymax);
    }

    return cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
}

float FaceMotionSegImpl::calcDynamicThreshold(const cv::Mat& W, const cv::Rect rect)
{
    float w_sum = 0;
    int r, c, samples_count = 0;
    std::vector<float> samples(rect.area());
    cv::Point2i tl = rect.tl(), br = rect.br();
    float* W_data = nullptr;
    int delta = 0;
    for (r = tl.y; r < br.y; ++r)
    {
        W_data = ((float*)W.data) + r*W.cols + tl.x;
        for (c = tl.x; c < br.x; ++c)
        {
            if (*W_data > 0)
                samples[samples_count++] = *W_data;
            ++W_data;
        }
    }
    std::nth_element(samples.begin(), samples.begin() + samples_count / 3,
        samples.begin() + samples_count);
    float median = samples[samples_count / 3];

    return median;
    //return (w_sum / samples_count);
}

cv::Mat FaceMotionSegImpl::calcSegmentation(const cv::Mat& frame_color, const cv::Mat& W,
    const cv::Rect rect, const cv::Point2f& center, float t)
{
    cv::Mat seg = cv::Mat::zeros(W.size(), CV_8U);

    float motion_sum = 0;
    int r, c;
    cv::Point2i tl = rect.tl(), br = rect.br();
    float* W_data = nullptr;
    //float w_y, max_w_y = (float)br.y - center.y;
    unsigned char* seg_data = nullptr;
    int delta = 0;
    for (r = tl.y; r < br.y; ++r)
    {
        delta = r*seg.cols + tl.x;
        W_data = ((float*)W.data) + delta;
        seg_data = seg.data + delta;
        //w_y = 1.0f - (std::max(0.0f, (float)r - center.y) / max_w_y);
        for (c = tl.x; c < br.x; ++c)
        {
            *seg_data++ = (*W_data++ > t) ? 128 : 0;
            //            *seg_data++ = ((*W_data++)*w_y > 4.0f) ? 128 : 0;
        }
    }
    /*
    cv::morphologyEx(seg, seg, cv::MorphTypes::MORPH_CLOSE,
    cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(5, 5)),
    cv::Point(-1, -1), 5);
    */
    return seg;
}

cv::Mat FaceMotionSegImpl::calcSegmentation_2(const cv::Mat& frame_color, const cv::Mat& W,
    const cv::Rect rect, const cv::Point2f& center, float t)
{
    int r, c;
    cv::Point2i tl = rect.tl(), br = rect.br();
    cv::Mat seg = cv::Mat::zeros(W.size(), CV_8U);
    cv::Mat markers = cv::Mat::zeros(W.size(), CV_32S);
    float* W_data = nullptr;
    //float w_y, max_w_y = (float)br.y - center.y;

    // Process rect pixels
    for (r = tl.y; r < br.y; ++r)
    {
        W_data = ((float*)W.data) + r*W.cols + tl.x;
        //w_y = 1.0f - (std::max(0.0f, (float)r - center.y) / max_w_y);
        for (c = tl.x; c < br.x; ++c)
        {
            if (*W_data++ > t)
                //if ((*W_data++)*w_y > 4.0f)
                markers.at<int>(r, c) = 2;
        }
    }

    ///
    // Process border
    ///
    float curr_value;

    // top
    r = tl.y;
    for (c = tl.x; c < br.x; ++c)
    {
        markers.at<int>(r, c) = 1;
    }
    /*
    // bottom
    r = br.y - 1;
    for (c = tl.x; c < br.x; ++c)
    {
    markers.at<int>(r, c) = 1;
    }
    */

    // left
    c = c = tl.x;
    for (r = tl.y; r < br.y; ++r)
    {
        markers.at<int>(r, c) = 1;
    }

    // right
    c = br.x - 1;
    for (r = tl.y; r < br.y; ++r)
    {
        markers.at<int>(r, c) = 1;
    }

    //
    cv::Mat W_gray, W_color;
    W.convertTo(W_gray, CV_8U);
    cv::cvtColor(W_gray, W_color, cv::COLOR_GRAY2BGR);

    cv::watershed(W_color, markers);

    // Output segmentation
    int* markers_data = nullptr;
    unsigned char* seg_data = nullptr;
    int delta;
    for (r = tl.y; r < br.y; ++r)
    {
        delta = r*markers.cols + tl.x;
        markers_data = ((int*)markers.data) + delta;
        seg_data = ((unsigned char*)seg.data) + delta;
        for (c = tl.x; c < br.x; ++c)
        {
            *seg_data++ = (*markers_data++ == 2) ? 128 : 0;
        }
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(seg, seg, kernel, cv::Point(-1, -1), 5);

    return seg;
}

void FaceMotionSegImpl::removeNeck(cv::Mat& seg, const cv::Rect& rect,
    const dlib::full_object_detection& landmarks)
{
    cv::Point2i tl = rect.tl(), br = rect.br();

    // Extract relevant points for the left side of the jaw
    cv::Point2f p4(landmarks.part(4).x(), landmarks.part(4).y());
    cv::Point2f p5(landmarks.part(5).x(), landmarks.part(5).y());
    cv::Point2f p6(landmarks.part(6).x(), landmarks.part(6).y());

    // Extract relevant points for the right side of the jaw
    cv::Point2f p10(landmarks.part(10).x(), landmarks.part(10).y());
    cv::Point2f p11(landmarks.part(11).x(), landmarks.part(11).y());
    cv::Point2f p12(landmarks.part(12).x(), landmarks.part(12).y());

    // Find the average direction for the left side
    cv::Point2f v4_5 = p4 - p5;
    cv::Point2f v5_6 = p5 - p6;
    v4_5 = v4_5 / cv::norm(v4_5);
    v5_6 = v5_6 / cv::norm(v5_6);
    cv::Point2f vl = v4_5 + v5_6;
    vl *= ((tl.x - p4.x) / vl.x);

    // Find the average direction for the right side
    cv::Point2f v12_11 = p12 - p11;
    cv::Point2f v11_10 = p11 - p10;
    v12_11 = v12_11 / cv::norm(v12_11);
    v11_10 = v11_10 / cv::norm(v11_10);
    cv::Point2f vr = v12_11 + v11_10;
    vr *= ((br.x - p12.x) / vr.x);

    // Find the left and right edge points
    cv::Point2f pl = p4 + vl;
    cv::Point2f pr = p12 + vr;

    // Construct contour
    std::vector<std::vector<cv::Point>> contours(1);
    std::vector<cv::Point>& contour = contours[0];
    contour.resize(13);
    contour[0] = cv::Point((int)std::roundf(pl.x), (int)std::roundf(pl.y));
    contour[1] = cv::Point((int)p4.x, (int)p4.y);
    contour[2] = cv::Point((int)p5.x, (int)p5.y);
    contour[3] = cv::Point((int)p6.x, (int)p6.y);
    contour[4] = cv::Point(landmarks.part(7).x(), landmarks.part(7).y());
    contour[5] = cv::Point(landmarks.part(8).x(), landmarks.part(8).y());
    contour[6] = cv::Point(landmarks.part(9).x(), landmarks.part(9).y());
    contour[7] = cv::Point((int)p10.x, (int)p10.y);
    contour[8] = cv::Point((int)p11.x, (int)p11.y);
    contour[9] = cv::Point((int)p12.x, (int)p12.y);
    contour[10] = cv::Point((int)std::roundf(pr.x), (int)std::roundf(pr.y));
    contour[11] = br;
    contour[12] = cv::Point(tl.x, br.y);

    // Clear contour area
    cv::drawContours(seg, contours, 0, cv::Scalar(0, 0, 0), CV_FILLED);
}

void FaceMotionSegImpl::removeSecondaryObjects(cv::Mat& seg, const cv::Rect& rect,
    const std::vector<std::vector<cv::Point>>& contours)
{
    std::vector<double> contourAreas(contours.size());
    for (size_t i = 0; i < contours.size(); ++i)
        contourAreas[i] = cv::contourArea(contours[i]);
    int max_i = std::distance(contourAreas.begin(),
        std::max_element(contourAreas.begin(), contourAreas.end()));
    for (size_t i = 0; i < contours.size(); ++i)
    {
        if (i == max_i) continue;
        cv::drawContours(seg, contours, i, cv::Scalar(0, 0, 0), CV_FILLED);
    }
}

float FaceMotionSegImpl::calcScore(const cv::Mat& frame_color, const cv::Mat& W,
    const cv::Mat& seg, const cv::Rect rect)
{
    /*
    int r, c;
    cv::Point2i tl = rect.tl(), br = rect.br();
    float* W_data = nullptr;
    float rect_score = 0, img_score = 0;

    // Process rect pixels
    for (r = tl.y; r < br.y; ++r)
    {
    W_data = ((float*)W.data) + r*W.cols + tl.x;
    for (c = tl.x; c < br.x; ++c)
    {
    rect_score += *W_data++;
    }
    }

    // Process all pixels
    W_data = (float*)W.data;
    for (r = 0; r < W.rows; ++r)
    {
    for (c = 0; c < W.cols; ++c)
    {
    img_score += *W_data++;
    }
    }

    return (rect_score / img_score);

    */

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(seg.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return 0;

    std::vector<double> contourAreas(contours.size());
    for (size_t i = 0; i < contours.size(); ++i)
        contourAreas[i] = cv::contourArea(contours[i]);
    int max_i = std::distance(contourAreas.begin(),
        std::max_element(contourAreas.begin(), contourAreas.end()));
    std::vector<cv::Point>& max_contour = contours[max_i];
    if (contourAreas[max_i] <= 1) return 0;

    std::vector<std::vector<cv::Point>> hull(1);
    cv::convexHull(cv::Mat(contours[max_i]), hull[0], false);
    float convexity = float(contourAreas[max_i] / cv::contourArea(hull[0]));
    float proportion = std::min(float(contourAreas[max_i]) / (rect.area() * 0.25f), 1.0f);

    // Find max contour edge length
    std::vector<cv::Point> approx_max_contour;
    cv::approxPolyDP(max_contour, approx_max_contour, 2, true);

    float total_length = 0, edge_length;
    float avg_weighted_length = 0;
    for (size_t i = 1; i < max_contour.size(); ++i)
    {
        edge_length = (float)cv::norm(max_contour[i - 1] - max_contour[i]);
        avg_weighted_length += edge_length*edge_length;
        total_length += edge_length;
    }
    avg_weighted_length /= total_length;
    for (size_t i = 1; i <= approx_max_contour.size(); ++i)
        avg_weighted_length = std::max(avg_weighted_length,
        (float)cv::norm(approx_max_contour[i - 1] - approx_max_contour[i % approx_max_contour.size()]));
    float complexity = 1.0f - ((avg_weighted_length * avg_weighted_length) /
        (float)contourAreas[max_i]);
    /*
    float max_edge_length = 0;
    for (size_t i = 1; i <= approx_max_contour.size(); ++i)
        max_edge_length = std::max(max_edge_length,
        (float)cv::norm(approx_max_contour[i - 1] - approx_max_contour[i % approx_max_contour.size()]));
    float complexity = 1.0f - ((max_edge_length * max_edge_length * 2) /
        (float)contourAreas[max_i]);
        */
    complexity = std::max(0.0f, std::min(1.0f, complexity));
    /*
    float max_edge_length = 0;
    for (size_t i = 1; i < max_contour.size(); ++i)
        max_edge_length = std::max(max_edge_length, 
            (float)cv::norm(max_contour[i - 1] - max_contour[i]));

    float complexity = std::min(sqrt(((float)contourAreas[max_i] * 0.25f) / 
        (max_edge_length*max_edge_length)), 1.0f);
    */
    float score = (convexity + proportion + complexity) / 3;

    /// Debug ///
    if (verbose >= 2)
    {
        cv::Mat out(frame_color.clone());
        cv::drawContours(out, contours, max_i, cv::Scalar(0, 255, 0));
        cv::drawContours(out, hull, 0, cv::Scalar(0, 0, 255));
        std::vector<std::vector<cv::Point>> approx_contours = { approx_max_contour };
        cv::drawContours(out, approx_contours, 0, cv::Scalar(255, 0, 0));
        cv::putText(out, (boost::format("Convexity: %f") % convexity).str(), cv::Point(15, 15),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
        cv::putText(out, (boost::format("Proportion: %f") % proportion).str(), cv::Point(15, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
        cv::putText(out, (boost::format("Complexity: %f") % complexity).str(), cv::Point(15, 55),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
        cv::putText(out, (boost::format("Total Score: %f") % score).str(), cv::Point(15, 75),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, CV_AA);
        cv::imwrite(str(boost::format("%s\\score_%04d.jpg") %
            outputPath % frame_ids[cfi]), out);
        cv::imshow("score", out);
        cv::waitKey(1);
    }
    /////////////

    return score;
}

void FaceMotionSegImpl::renderLandmarks(const cv::Mat& img,
    const dlib::full_object_detection& landmarks,
    const cv::Scalar& color)
{
    if (landmarks.num_parts() != 68)
        throw std::runtime_error("Each shape size must be exactly 68!");

    const dlib::full_object_detection& d = landmarks;
    for (unsigned long i = 1; i <= 16; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);

    for (unsigned long i = 28; i <= 30; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);

    for (unsigned long i = 18; i <= 21; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);
    for (unsigned long i = 23; i <= 26; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);
    for (unsigned long i = 31; i <= 35; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);
    cv::line(img, cv::Point(d.part(30).x(), d.part(30).y()),
        cv::Point(d.part(35).x(), d.part(35).y()), color);

    for (unsigned long i = 37; i <= 41; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);
    cv::line(img, cv::Point(d.part(36).x(), d.part(36).y()),
        cv::Point(d.part(41).x(), d.part(41).y()), color);

    for (unsigned long i = 43; i <= 47; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);
    cv::line(img, cv::Point(d.part(42).x(), d.part(42).y()),
        cv::Point(d.part(47).x(), d.part(47).y()), color);

    for (unsigned long i = 49; i <= 59; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);
    cv::line(img, cv::Point(d.part(48).x(), d.part(48).y()),
        cv::Point(d.part(59).x(), d.part(59).y()), color);

    for (unsigned long i = 61; i <= 67; ++i)
        cv::line(img, cv::Point(d.part(i).x(), d.part(i).y()),
        cv::Point(d.part(i - 1).x(), d.part(i - 1).y()), color);
    cv::line(img, cv::Point(d.part(60).x(), d.part(60).y()),
        cv::Point(d.part(67).x(), d.part(67).y()), color);

    // Add labels
    for (unsigned long i = 0; i < 68; ++i)
        cv::putText(img, std::to_string(i), cv::Point(d.part(i).x(), d.part(i).y()),
        cv::FONT_HERSHEY_PLAIN, 0.5, color, 1.0);

}

cv::Mat FaceMotionSegImpl::renderWeightMap(const cv::Mat& weightMap)
{
    cv::Mat w_norm, w_color;
    cv::normalize(weightMap, w_norm, 0, 1, cv::NORM_MINMAX, CV_32F);
    w_norm.convertTo(w_color, CV_8U, 255.0);
    cv::applyColorMap(w_color, w_color, cv::COLORMAP_JET);
    return w_color;
}

void FaceMotionSegImpl::drawSegmentation(cv::Mat& frame_color, const cv::Rect& rect,
    const cv::Mat& seg)
{
    int r, c;
    float a;
    cv::Point2i tl = rect.tl(), br = rect.br();
    cv::Point3_<uchar>* img_data = nullptr;
    unsigned char* seg_data = nullptr;
    for (r = tl.y; r < br.y; ++r)
    {
        img_data = ((cv::Point3_<uchar>*)frame_color.data) + r*frame_color.cols + tl.x;
        seg_data = seg.data + r*seg.cols + tl.x;
        for (c = tl.x; c < br.x; ++c)
        {
            a = *seg_data++ / 255.0f;
            img_data->x = (unsigned char)std::round(0 * a + img_data->x*(1 - a));
            img_data->y = (unsigned char)std::round(0 * a + img_data->y*(1 - a));
            img_data->z = (unsigned char)std::round(255 * a + img_data->z*(1 - a));
            ++img_data;
        }
    }
}

void FaceMotionSegImpl::writeSegmentation(cv::Mat& frame_color, const cv::Rect& rect,
    const cv::Mat& seg, int frame_id)
{
    if (outputPath.empty()) return;

    // Convert segmentation to color image
    cv::Mat seg_color = cv::Mat::zeros(seg.size(), CV_8UC3);
    int r, c, delta;
    cv::Point2i tl = rect.tl(), br = rect.br();
    cv::Point3_<uchar>* seg_color_data = nullptr;
    unsigned char* seg_data = nullptr;
    for (r = tl.y; r < br.y; ++r)
    {
        delta = r*seg.cols + tl.x;
        seg_color_data = ((cv::Point3_<uchar>*)seg_color.data) + delta;
        seg_data = seg.data + delta;
        for (c = tl.x; c < br.x; ++c)
        {
            if (*seg_data++ > 0)
                *seg_color_data = cv::Point3_<uchar>(128, 128, 192);  // Pink
            ++seg_color_data;
        }
    }

    // Crop segmentation and corresponding frame
    cv::Mat frame_cropped = frame_color(rect);
    cv::Mat seg_cropped = seg_color(rect);
    cv::MatSize size = frame_cropped.size;
    int max_index = std::distance(size.p, std::max_element(size.p, size.p + 2));
    if (size[max_index] > 500)
    {
        float scale = 500.0f / (float)size[max_index];
        int w = (int)std::round(frame_cropped.cols * scale);
        int h = (int)std::round(frame_cropped.rows * scale);
        cv::resize(frame_cropped, frame_cropped, cv::Size(w, h));
        cv::resize(seg_cropped, seg_cropped, cv::Size(w, h));
    }

    //cv::resize(frame_cropped, frame_cropped, cv::Size(500, 500));
    //cv::resize(seg_cropped, seg_cropped, cv::Size(500, 500));

    cv::imwrite(str(boost::format("%s\\frame_%04d.png") %
        outputPath % frame_id), frame_cropped);
    cv::imwrite(str(boost::format("%s\\seg_%04d.png") %
        outputPath % frame_id), seg_cropped);
}

}   // namespace motionseg
