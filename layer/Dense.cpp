#include "Dense.h"

using namespace std;

Dense::Dense(int p_input_len, int p_nb_node, std::string p_activation): nb_node(p_nb_node)
{
    for (int i = 0; i < p_nb_node; ++i)
    {
        Neuron x = Neuron(p_input_len, p_activation);
        x.init("heInit");
        nodes.push_back(x);
    }

}


Dense::Dense(const Dense& dense): nb_node(dense.nb_node), nodes(dense.nodes)
{
}


Dense::Dense(): nb_node(1)
{
    for (int i = 0; i < nb_node; ++i)
        nodes.push_back(Neuron(128, "sigmoid"));

}


void Dense::init(std::string initializer)
{
    /*for (Neuron node : nodes)
    {
        //node->init(initializer);
    }*/
}



std::vector<double> Dense::forward(std::vector<double> x) const
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




pair<vector<vector<double>>, vector<vector<double>>> Dense::update(vector<double> grad, float alpha, vector<double> x, vector<double> y, float moment)
{
    pair<vector<double>, vector<double>> result;

    vector<vector<double>> error_mat(nb_node, vector<double>(nodes[0].input_len, 0.0));
    vector<vector<double>> new_grad(nb_node, vector<double>(nodes[0].input_len, 0.0));

    vector<std::vector<double>> prev_grad(nb_node, vector<double>(nodes[0].input_len, 0.0));

    for (int i = 0; i < nb_node; i++)
    {
        result = nodes[i].update(grad[i], alpha, x, y[i], moment, prev_grad[i]);
        error_mat[i] = result.first;
        new_grad[i] = result.second;
    }

    return make_pair(error_mat, new_grad);
}



pair<vector<vector<double>>, vector<vector<double>>> Dense::update(vector<double> grad, float alpha, vector<double> x, vector<double> y, float moment, vector<vector<double>> prev_grad)
{
    pair<vector<double>, vector<double>> result;

    vector<vector<double>> error_mat(nb_node, vector<double>(nodes[0].input_len, 0.0));
    vector<vector<double>> new_grad(nb_node, vector<double>(nodes[0].input_len, 0.0));

    for (int i = 0; i < nb_node; i++)
    {
        result = nodes[i].update(grad[i], alpha, x, y[i], moment, prev_grad[i]);
        error_mat[i] = result.first;
        new_grad[i] = result.second;
    }

    return make_pair(error_mat, new_grad);
}


Dense::~Dense(){}


