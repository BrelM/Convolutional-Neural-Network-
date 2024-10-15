
#include "reader.h"


std::vector<std::vector<double>> read_image(const std::string& path)
{

    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);

    if(image.empty())
    {
        std::cerr << "Failed to read the image." << path << std::endl;
        return std::vector<std::vector<double>>();
    }

    cv::Mat resizedImage;

    std::vector<cv::Rect> faces = detectFace(image);
    for(cv::Rect face : faces)
    {
        image = image(cv::Range(face.x, face.x + face.width), cv::Range(face.y, face.y + face.height)).clone();
    }

    cv::resize(image, resizedImage, cv::Size(224, 224));


    //cv::imshow(string("Detected Faces"), resizedImage);
    //cv::waitKey(0);


    std::vector<std::vector<double>> matrix;
    for(int row = 0; row < resizedImage.rows; ++row)
    {
        std::vector<double> rowData;
        for(int col = 0; col < resizedImage.cols; ++col)
        {
            rowData.push_back(static_cast<double>(resizedImage.at<uchar>(row, col)));
        }
        matrix.push_back(rowData);
    }

    return matrix;
}


cv::Mat read_as_cvmat(const std::string& path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);

    if(image.empty())
    {
        std::cerr << "Failed to read the image." << path << std::endl;
        return cv::Mat();
    }

    cv::Mat resizedImage;

    std::vector<cv::Rect> faces = detectFace(image);
    for(cv::Rect face : faces)
    {
        image = image(cv::Range(face.x, face.x + face.width), cv::Range(face.y, face.y + face.height)).clone();
    }

    cv::resize(image, resizedImage, cv::Size(224, 224));

    //cv::imshow(string("Detected Faces"), resizedImage);
    //cv::waitKey(0);

    return resizedImage;
}






vector<cv::Rect> detectFace(const cv::Mat& image)
{
    cv::CascadeClassifier faceCascade;

    //faceCascade.load("https://github.com/kipr/opencv/blob/31450d613c0c091c6ad510cf2a42a25edbe01e62/data/haarcascades/haarcascade_frontalface_alt2.xml");
    faceCascade.load("haarcascade_frontalface_default.xml");

    cv::Mat grayImage = image;
    //cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0, cv::Size(0, 0), cv::Size(0, 0));

    return faces;
}





vector<vector<double>> to_vector(cv::Mat& image)
{
    vector<vector<double>> matrix;
    for(int row = 0; row < image.rows; ++row)
    {
        std::vector<double> rowData;
        for(int col = 0; col < image.cols; ++col)
        {
            rowData.push_back(static_cast<double>(image.at<uchar>(row, col)));
        }
        matrix.push_back(rowData);
    }

    return matrix;
}


