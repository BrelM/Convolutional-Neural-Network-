

#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <filesystem>

#include <opencv2/opencv.hpp>


// #include <opencv2/opencv.hpp>  // Uncomment this line if using OpenCV for image processing

#include "reader.h"
#include "function.h"

namespace fs = filesystem;
using namespace std;



class Dataframe
{
public:
    vector<vector<vector<double>>> items = {};
    vector<vector<double>> labels = {};
    vector<int> shape = {};

    Dataframe(const string &path = "");

    void set_shape();

    void drop();

    vector<vector<double>>&  operator[](int idx);

    friend ostream& operator<<(ostream& os, const Dataframe& df);

};




class DataframeTest
{
public:
    vector<cv::Mat> items;
    vector<vector<double>> labels = {};
    vector<int> shape = {};

    DataframeTest(const string &path = "");

    void set_shape();

    void drop();

    cv::Mat   operator[](int idx);

    friend ostream& operator<<(ostream& os, const DataframeTest& df);

};



#endif
