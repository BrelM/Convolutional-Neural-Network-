#ifndef IMAGE_READER_H
#define IMAGE_READER_H

#include <vector>
#include <cstdint>
#include <iostream>
#include <string.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

/*
struct PNGImage
{
    vector<vector<uint8_t>> matrix;
    uint32_t width;
    uint32_t height;
};
*/

vector<vector<double>> read_image(const string& path);

cv::Mat read_as_cvmat(const std::string& path);

vector<cv::Rect> detectFace(const cv::Mat& image);

vector<vector<double>> to_vector(cv::Mat& image);

#endif // IMAGE_READER_H
