#ifndef OUTPUT_H
#define OUTPUT_H

#include "../nn/neuron_output.h"


class Output
{
public:

    int nb_node;
    std::vector<NeuronOutput> nodes;



    Output();
    Output(const Output&);
    Output(int input_len, int nb_node, std::string activation="sigmoid");

    ~Output();

    void init(std::string initializer);

    std::vector<double> forward(std::vector<double> x) const;

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> update(const std::vector<double> grad, float alpha, const std::vector<double> y, float moment);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> update(const std::vector<double> grad, float alpha, const std::vector<double> y, float moment, const std::vector<std::vector<double>> prev_grad);


};

#endif /* OUTPUT_H */
