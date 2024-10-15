#include "Output.h"
#include <iostream>

Output::Output(int p_input_len, int p_nb_node, std::string p_activation): nb_node(p_nb_node)
{
    for (int i = 0; i < p_nb_node; ++i)
    {
        NeuronOutput x = NeuronOutput(p_input_len, p_activation);
        x.init("heInit");
        nodes.push_back(x);
    }

}


Output::Output(const Output& output): nb_node(output.nb_node), nodes(output.nodes)
{
}


Output::Output(): nb_node(1)
{
    for (int i = 0; i < nb_node; ++i)
        nodes.push_back(NeuronOutput(128, "sigmoid"));

}


void Output::init(std::string initializer)
{
    /*for (NeuronOutput node : nodes)
    {
        //node->init(initializer);
    }*/
}



std::vector<double> Output::forward(std::vector<double> x) const
{
    //cout << "output" << endl;

    std::vector<double> y(nb_node, 0.0);
    for (int node = 0; node < nb_node; node++)
    {
        double z = nodes[node].net(x);

        //if(nodes[node].activation == "sigmoid")
        y[node] = sigmoid(z);

    }

    return y;
}




//Update with no previous gradient
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> Output::update(const std::vector<double> grad, float alpha, const std::vector<double> y, float moment)
{
    //std::cout << "upd_out_empty "<< nodes[0].weights.size() << std::endl;
    std::pair<std::vector<double>, std::vector<double>> result;

    std::vector<std::vector<double>> error_mat(nb_node, std::vector<double>(nodes[0].input_len, 0.0)), new_grad(nb_node, std::vector<double>(nodes[0].input_len, 0.0)), prev_grad(nb_node, std::vector<double>(nodes[0].input_len, 0.0));

    for (int i = 0; i < nb_node; ++i)
    {
        result = nodes[i].update(grad[i], alpha, y, moment, prev_grad[i]);

        error_mat[i] = result.first;
        new_grad[i] = result.second;
    }

    //std::cout << "upd_out_empty "<< nodes[0].weights.size() << std::endl;
    return std::make_pair(error_mat, new_grad);
}




//Update with previous gradient
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> Output::update(const std::vector<double> grad, float alpha, const std::vector<double> y, float moment, const std::vector<std::vector<double>> prev_grad)
{
    //std::cout << "upd_out_filled "<< nodes[0].weights.size() << std::endl;
    std::pair<std::vector<double>, std::vector<double>> result;

    std::vector<std::vector<double>> error_mat(nb_node, std::vector<double>(nodes[0].input_len, 0.0));
    std::vector<std::vector<double>> new_grad(nb_node, std::vector<double>(nodes[0].input_len, 0.0));

    for (int i = 0; i < nb_node; ++i)
    {
        result = nodes[i].update(grad[i], alpha, y, moment, prev_grad[i]);

        error_mat[i] = result.first;
        new_grad[i] = result.second;
    }

    //std::cout << "upd_out_filled_end "<< nodes[0].weights.size() << std::endl;
    return std::make_pair(error_mat, new_grad);
}





Output::~Output(){}







