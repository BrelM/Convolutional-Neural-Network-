/*
    neuron_output.cpp

    Class NeuronOutput
    An Abstract class for a node in standard neural networks .
    Its purpose is to serve as a node to be added in Ouput layers.

*/
#include "neuron_output.h"

NeuronOutput::NeuronOutput(const int input_l, std::string activ): input_len(input_l+1), activation(activ) //Constructor
{
    for(int i = 0; i < input_l; i++)
        weights.push_back(0);
}

void NeuronOutput::init(std::string initializer) // Initializer
{
    if(initializer=="heInit")
         weights = heInitNode(input_len);
}


double NeuronOutput::net(std::vector<double> x) const// weighted sum of inputs with table
{
    //std::cout << "******** "<< x.size() << " ******** "<< weights.size();
    int a = input_len;
    if(static_cast<int>(x.size()) != a - 1)
        throw std::runtime_error("The input and the weight sizes don't match."" Sizes are " + std::to_string(x.size()) + " and " + std::to_string(a-1));

    double sum = 0;
    for(size_t i = 0; i < x.size(); i++)
        sum += (x[i] * weights[i]);

    return sum- weights.back();
}





std::pair<std::vector<double>, std::vector<double>> NeuronOutput::update(double grad, float alpha, vector<double> y, float moment, vector<double> prev_grad)  // Weights updater
{
    std::vector<double> weighted_error, new_grad;

    y.push_back(-1);

    for(int i = 0; i < input_len; i++)
    {
        weighted_error.push_back(grad * weights[i]);
        new_grad.push_back(static_cast<double>(alpha * grad * y[i] + moment * prev_grad[i]));
        weights[i] += new_grad.back();
    }
    weighted_error.pop_back();

    return std::make_pair(weighted_error, new_grad);
}

