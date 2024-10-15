/*
    neuron_output.h

    Class NeuronOutput
    An Abstract class for a node in standard neural networks .
    Its purpose is to serve as a node to be added in Output layers.

*/

#ifndef NEURON_OUTPUT_H
#define NEURON_OUTPUT_H

// Importing built-in support functions
#include <iostream>
#include <vector>
#include <string>


#include "../activation_function/activation_function.h"
#include "../utils/initializer.h"
#include "../utils/function.h"


// Importing our support functions

class NeuronOutput
{
    public:

        std::vector<double> weights;
        int input_len;
        std::string activation;


        NeuronOutput(const int, std::string); //Constructor

        void init(std::string initializer="heInit"); // Initializer
        double net(std::vector<double> x) const; // weighted sum of inputs with vector

        std::pair<std::vector<double>, std::vector<double>> update(double grad, float alpha, std::vector<double> y, float moment, std::vector<double> prev_grad);  // Weights updater

        std::pair<std::vector<double>, std::vector<double>> update(double grad, float alpha, std::vector<double> y, float moment);  // Weights updater


};



#endif
