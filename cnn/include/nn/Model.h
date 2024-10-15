/*
    Model.h

    Class Model
    An Abstract class for a standard neural network template.
    Its purpose is to serve as the base in which will be added layers.

*/

#ifndef MODEL_H
#define MODEL_H

// Importing built-in support functions
#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#include <vector>
#include <tuple>

// Importing layers
#include "../layer/Layer.h"
#include "../layer/Conv.h"
#include "../layer/Pool.h"
#include "../layer/Dense.h"
#include "../layer/Flatten.h"
#include "../layer/Output.h"

// Importing our support functions
#include "../utils/function.h"
#include "../utils/gradient.h"
#include "../utils/loss.h"




class Model
{

    public:
        std::vector<Conv*> conv_layers = {};
        std::vector<Pool*> pool_layers = {};
        std::vector<Dense*> dense_layers = {};
        Output* output_layer = 0;
        std::vector<int> layers_order = {};
        int nb_kernel = 2;



        Model(const int nb_kerne=1); //Constructor declaration but with layers as parameters
        Model(const Model&); //Constructor declaration but with layers as parameters

        ~Model(); // Destructor


        void addLayer(Conv);
        void addLayer(Pool);
        void addLayer(Dense);
        void addLayer(Output);


        void init(const int nb_kernel=1, const int conv_shape=3, const int conv_stride=1, const int pool_shape=3, const int pool_stride=1, const int padding=1); // Model's layers initializer declaration

        void train(const std::vector<std::vector<std::vector<double>>> x_train, const std::vector<std::vector<double>> y_train, const float alpha = 0.2, const int epochs=500, int batch_size=0, const float moment=0.001, const float threshold=0.001); // Model's train method declaration

        std::tuple<std::vector<std::vector<std::vector<std::vector<double>>>>, std::vector<std::vector<double>>> forward(std::vector<std::vector<double>> x) const; // Model's forward method declaration

        std::vector<vector<vector<double>>> backward(std::vector<double> pred, std::vector<std::vector<double>> input, std::vector<std::vector<double>> outputs_dense, std::vector<std::vector<std::vector<std::vector<double>>>> outputs_conv, const std::vector<double> y_train, const float alpha, const float moment, std::vector<std::vector<std::vector<double>>> prev_grad); // Model's backward method declaration

        std::vector<double> predict(const std::vector<std::vector<double>> x) const; // Model's predict method declaration


};



#endif
