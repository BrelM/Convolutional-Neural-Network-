#include <iostream>
#include <cmath>

#include "activation_function.h"


using namespace std;
double sigmoid(double net_input){
    return 1 / (1 + exp(-net_input));
}

double ReLU(double x) {
    return x >= 0 ? x : x * 1e-3;
}

double LReLu(double x) {
    return std::max(x, 0.0);
}
