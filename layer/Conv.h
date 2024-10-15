#ifndef CONV_H
#define CONV_H


#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <type_traits>

#include "../utils/initializer.h"
#include "../utils/function.h"
#include "../activation_function/activation_function.h"

using namespace std;

class Conv
{
public:
    int nb_kernel;
    int shape;
    int stride;
    int padding;
    vector<vector<vector<double>>> kernel;


    Conv();

    Conv(const Conv&);

    Conv(int p_nb_kernel, int p_shape, int p_stride, int p_padding);


    ~Conv();

    void init(int p_nb_kernel, int p_shape, int p_stride, int p_padding);

    vector<vector<vector<double>>> forward(const vector<vector<double>> x) const;


    vector<vector<vector<double>>> forward(const vector<vector<vector<double>>> x) const;


    vector<vector<vector<double>>> update(const vector<vector<vector<double>>> &gradients,
                                                        vector<vector<vector<double>>> &inputs0,
                                                        vector<vector<vector<double>>> &inputs1,
                                                         double alpha);


    vector<vector<vector<double>>> update(const vector<vector<vector<double>>> &gradients,
                                                        vector<vector<vector<double>>> &inputs0,
                                                        vector<vector<double>> &inputs1,
                                                        double alpha);


    void update(const vector<vector<vector<double>>> &gradients,
                                                        vector<vector<double>> &inputs0,
                                                         double alpha);


};

#endif // CONV_H
