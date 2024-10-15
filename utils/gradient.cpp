

#include <iostream>
#include <vector>

#include "gradient.h"

using namespace std ;

vector<double> gradient(vector<double> output, vector<double> classes)
{
    vector<double> b;

    for (size_t i = 0; i < output.size(); i++) {
        b.push_back((classes[i] - output[i]) * output[i] * (1 - output[i]));
    }

    return b;
}
