#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include <pickle.h>


#include "utils/Dataframe.h"
#include "nn/Model.h"
#include "layer/Conv.h"
#include "layer/Pool.h"
#include "layer/Dense.h"
#include "layer/Output.h"
#include "utils/reader.h"

using namespace std;
using namespace cv;


vector<double> unknown_person = {0.0, 0.0, 0.0, 0.0};
vector<double> brad = {0.0, 0.0, 0.0, 1.0};
vector<double> jolie = {0.0, 1.0, 0.0, 0.0};
vector<double> dicaprio = {1.0, 0.0, 0.0, 0.0};


int main()
{

	string train_path = "train_data/";
	string test_path = "test_data/";

	Dataframe train_data(train_path);
	DataframeTest test_data(test_path);

	cout << train_data.shape[0] << endl;
	cout << test_data.shape[0] << endl;

	//vector<string> labels = {"Jolie", "Dicaprio", "Brad", "Unknown"};

	int nb_kernel = 2;

	Model model;
	model.addLayer(Conv(nb_kernel, 3, 3, 0));
	model.addLayer(Pool(3, 1));
	model.addLayer(Conv(nb_kernel, 3, 3, 0));
	model.addLayer(Pool(3, 1));
	model.addLayer(Conv(nb_kernel, 3, 3, 0));
	model.addLayer(Dense(128, 2, "sigmoid"));
	model.addLayer(Output(2, 4, "sigmoid"));

	model.init(nb_kernel, 3, 3, 3, 1, 0);


	cout << "Training" << endl;
	model.train(train_data.items, train_data.labels, 1e-1, 10000, 0, 0.001, 0.000007);

	ofstream model_file("model_file.md", ios::binary);

	usa::Pickle().dump(string("model_file"), model);

	model_file.close();

	ifstream model_file_load(string("model_file.md"), ios::binary);

	Model loaded_model;
	usa::Pickle().load(string("model_file"), loaded_model);
	model_file_load.close();

	//vector<function<void()>> choices = {loaded_model, model};
	vector<Model> choices = {loaded_model, model};
	cout << "What do you want to do?\n\t1. Load saved model, 0. Train model\n\t...";
	int c;
	cin >> c;
	Model selected_model = choices[c];

	//Model selected_model = model;
	for (int i = 0; i < test_data.shape[0]; i++)
	{
	    vector<vector<double>> to_send = to_vector(test_data.items[i]);

		cout << "Prediction of image of class '";

        if(test_data.labels[i] == jolie)
            cout << "Jolie";
        else if(test_data.labels[i] == brad)
            cout << "Brad";
        else if(test_data.labels[i] == dicaprio)
            cout << "Dicaprio";
        else if(test_data.labels[i] == unknown_person)
            cout << "Unknown";

        cout << "' gives '";
		vector<double> pred = selected_model.predict(to_send);

        if(pred == jolie)
            cout << "Jolie";
        else if(pred == brad)
            cout << "Brad";
        else if(pred == dicaprio)
            cout << "Dicaprio";
        else if(pred == unknown_person)
            cout << "Unknown";


		cout<< "'" << endl;
	}

    /*
    VideoCapture cap;
    cap.open(0, CAP_ANY);

    if(!cap.isOpened())
    {
        cout << "Error playing video" << endl;
        return -1;;
    }


    Scalar font_color(0, 255, 0);
    int thicknesss = 2;
    int font_size = 1;
    int font_weight = thicknesss;

    Mat frame;

    int i = -1;
    for(;;)
    {
        cap.read(frame);

        //frame.copyTo(frame(roi).clone());

        if(frame.empty())
            break;
        vector<Rect> bounds = detectFace(frame);
        Mat to_print = frame.clone();

        //Prediction here

        vector<std::string> labels = {"Unknown"};
        for(Rect bound: bounds)
        {
            Point text_position(bound.x, bound.y - font_size - 2);

            putText(to_print, labels[0], text_position, FONT_HERSHEY_COMPLEX_SMALL, font_size, font_color, font_weight);
            rectangle(to_print, bound, font_color, 2);
        }
        //Saving samples
        //Mat to_save = frame(Range(bounds[0].x, bounds[0].x+bounds[0].width), Range(bounds[0].y, bounds[0].y+bounds[0].height)).clone();

        //imwrite("samples/human"+std::to_string(i++)+".jpg", to_save);

        imshow("Live", frame);

        if(waitKey(5) >= 25)
            break;
    }*/

	return 0;
}
