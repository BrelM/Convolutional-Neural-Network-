#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include <string>
#include <utility>
#include <iostream>

#include "../nn/neuron.h"
#include "../utils/function.h"
#include "../activation_function/activation_function.h"

using namespace std;

class Dense
{
public:
    int nb_node;
    vector<Neuron> nodes;

    Dense(int p_input_len, int p_nb_node, string p_activation="sigmoid");

    Dense(const Dense&);

    Dense();

    ~Dense();

    void init(string initializer);

    vector<double> forward(vector<double> x) const;

    pair<vector<vector<double>>, vector<vector<double>>> update(vector<double> grad, float alpha, vector<double> x, vector<double> y, float moment);
    pair<vector<vector<double>>, vector<vector<double>>> update(vector<double> grad, float alpha, vector<double> x, vector<double> y, float moment, vector<vector<double>> prev_grad);

};

#endif // DENSE_H
