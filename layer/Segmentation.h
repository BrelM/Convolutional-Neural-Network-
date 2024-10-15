#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>

// cette fonction renvoie un tableau de ROI, 
std::vector<cv::Mat> detectAndExtractFaces(const cv::Mat &image);

#endif // SEGMENTATION_H
