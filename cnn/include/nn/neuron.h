/*
    Neuron.h

    Class Neuron
    An Abstract class for a node in standard neural networks .
    Its purpose is to serve as a node to be added in Dense layers.

*/

#ifndef NEURON_H
#define NEURON_H

// Importing built-in support functions
#include <iostream>
#include <vector>
#include <string>
#include <tuple>


// Importing our support functions
#include "../activation_function/activation_function.h"
#include "../utils/initializer.h"
#include "../utils/function.h"


class Neuron
{
    public:
        vector<double> weights = {};
        int input_len;
        std::string activation;


        Neuron(const int, std::string); //Constructor

        void init(std::string initializer="heInit"); // Initializer
        double net(std::vector<double> x) const; // weighted sum of inputs with vector
        std::pair<std::vector<double>, std::vector<double>> update(double grad, float alpha, vector<double> x, double y, float moment, vector<double> prev_grad);  // Weights updaterc

};





#endif
