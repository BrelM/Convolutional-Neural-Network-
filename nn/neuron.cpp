/*
    Neuron.cpp

    Class Neuron
    An Abstract class for a node in standard neural networks .
    Its purpose is to serve as a node to be added in Dense layers.

*/

#include "neuron.h"


Neuron::Neuron(const int input_l, std::string activ): input_len(input_l+1), activation(activ) //Constructor
{
    for(int i = 0; i < input_l; i++)
        weights.push_back(0);
}




void Neuron::init(std::string initializer) // Initializer
{
    if(initializer=="heInit")
         weights = heInitNode(static_cast<int>(input_len));
}


double Neuron::net(std::vector<double> x) const // weighted sum of inputs with table
{
    int a = input_len;
    int len_x = static_cast<int>(x.size());
    if(len_x != a - 1)
        throw std::runtime_error("The input and the weight sizes don't match."" Sizes are " + std::to_string(x.size()) + " and " + std::to_string(a-1));


    double sum = 0;
    for(int i = 0; i < len_x; i++)
        sum += x[i] * weights[i];

    return sum - weights[weights.size()-1];
}


std::pair<std::vector<double>, std::vector<double>> Neuron::update(double grad, float alpha, vector<double> x, double y, float moment, vector<double> prev_grad)  // Weights updaterc
{

    std::vector<double> weighted_error, new_grad;

    x.push_back(-1);

    for(int i = 0; i < input_len; i++)
    {
        weighted_error.push_back(grad * weights[i]);
        new_grad.push_back(static_cast<double>(alpha * x[i] * grad * y * (1 - y) + moment * prev_grad[i]));
        weights[i] += new_grad.back();
    }
    weighted_error.pop_back();

    return std::make_pair(weighted_error, new_grad);
}
