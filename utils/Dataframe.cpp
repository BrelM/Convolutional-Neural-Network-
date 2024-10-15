
#include "Dataframe.h"
#include <vector>


Dataframe::Dataframe(const string &path)
{

    std::vector<double> unknown_person = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> brad = {0.0, 0.0, 0.0, 1.0};
    std::vector<double> jolie = {0.0, 1.0, 0.0, 0.0};
    std::vector<double> dicaprio = {1.0, 0.0, 0.0, 0.0};


    if (path.empty() || !fs::is_directory(path))
    {
        throw invalid_argument("Argument 1 must be the dataset folder path, relative or absolute.");
    }

    for (const auto &entry : fs::directory_iterator(path))
    {
        //if (entry.path().extension() == ".png")
        //{
            string file_path = entry.path().string();

            try
            {
                vector<vector<double>> file = read_image(file_path);
                // file = resize::resize_image(file);  // Uncomment this line if using resize functionality
                items.push_back(file);

                if(file_path.find("brad") != string::npos)
                {
                    labels.push_back(brad);
                }
                else if(file_path.find("jolie") != string::npos)
                {
                    labels.push_back(jolie);
                }
                else if(file_path.find("dicaprio") != string::npos)
                {
                    labels.push_back(dicaprio);
                }
                else
                {
                    labels.push_back(unknown_person);
                }
            }
            catch (const exception &e)
            {
                cout << "Could not load the file '" << file_path << "': " << e.what() << endl;
            }
        //}
    }

    set_shape();
}


void Dataframe::set_shape()
{
    shape = {static_cast<int>(items.size()), static_cast<int>(items[0].size()), static_cast<int>(items[0][0].size())};
}

void Dataframe::drop()
{
    // Implement the drop function logic here
}


vector<vector<double>>& Dataframe::operator[](int idx)
{
    return items[idx];
}


ostream& operator<<(ostream& os, const Dataframe& df)
{
    os << "Dataframe\n\n";
    os << "    (" << df.items.size() << " x " << df.items[0][0].size() << " x " << df.items[0].size() << ")\n\n";
    os << df.items[0][0][0] << "\n";
    os << "    \n";
    os << "    \n";
    os << "    ...\n";
    os << "    \n";
    os << "    \n";
    os << df.items.back().back().back() << "\n";
    os << "    labels: ";
    /*for (int label : df.labels)
    {
        os << label << " ";
    }*/
    os << "\n";

    return os;
}





DataframeTest::DataframeTest(const string &path)
{


    std::vector<double> unknown_person = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> brad = {0.0, 0.0, 0.0, 1.0};
    std::vector<double> jolie = {0.0, 1.0, 0.0, 0.0};
    std::vector<double> dicaprio = {1.0, 0.0, 0.0, 0.0};


    if (path.empty() || !fs::is_directory(path))
    {
        throw invalid_argument("Argument 1 must be the dataset folder path, relative or absolute.");
    }

    for (const auto &entry : fs::directory_iterator(path))
    {
        //if (entry.path().extension() == ".png")
        //{
            string file_path = entry.path().string();

            try
            {
                cv::Mat file = read_as_cvmat(file_path);

                // file = resize::resize_image(file);  // Uncomment this line if using resize functionality
                items.push_back(file);

                if(file_path.find("brad") != string::npos)
                {
                    labels.push_back(brad);
                }
                else if(file_path.find("jolie") != string::npos)
                {
                    labels.push_back(jolie);
                }
                else if(file_path.find("dicaprio") != string::npos)
                {
                    labels.push_back(dicaprio);
                }
                else
                {
                    labels.push_back(unknown_person);
                }
            }
            catch (const exception &e)
            {
                cout << "Could not load the file '" << file_path << "': " << e.what() << endl;
            }
        //}
    }

    set_shape();
}


void DataframeTest::set_shape()
{
    shape = {static_cast<int>(items.size()), static_cast<int>(items[0].size().height), static_cast<int>(items[0].size().width)};
}

void DataframeTest::drop()
{
    // Implement the drop function logic here
}


cv::Mat DataframeTest::operator[](int idx)
{
    return items[idx];
}


ostream& operator<<(ostream& os, const DataframeTest& df)
{
    os << "Dataframe\n\n";
    os << "    (" << df.items.size() << " x " << df.items[0].size().height << " x " << df.items[0].size().width << ")\n\n";
    os << df.items[0].at<uchar>(0,0) << "\n";
    os << "    \n";
    os << "    \n";
    os << "    ...\n";
    os << "    \n";
    os << "    \n";
    os << df.items.back().at<uchar>(df.items[0].cols-1, df.items[0].rows-1) << "\n";
    os << "    labels: ";

    /*
    for (int label : df.labels)
    {
        os << label << " ";
    }
    */
    os << "\n";

    return os;
}



