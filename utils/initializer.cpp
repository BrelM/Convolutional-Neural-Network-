
#include <iostream>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>


#include "initializer.h"

using namespace std ;

vector<double> heInitNode(int num_inputs) {

    float std_dev = 2.4 / num_inputs;
    vector<double> result;

    random_device rd; //cree une instance de la classe random comme base de generation
    mt19937 gen(rd());//generateur de nombre pseudi aleatoire
    uniform_real_distribution<double> dist(-std_dev, std_dev);

    for (int i = 0; i < num_inputs; i++) {
        result.push_back(dist(gen));
    }

    return result;
}


vector<vector<double>> gaussianInitKernel(int n)
{
    double sigma = 1;
    int center = n / 2;
    vector<vector<double>> matrix;
    for (int i = 0; i < n; i++)
    {
        vector<double> a(n, 0.0);
        matrix.push_back(a);
    }

    double constant = 1.0 / (2.0 * M_PI * sigma * sigma);

    for (int i = 0; i < n; i++) {
        vector<double> row;
        for (int j = 0; j < n; j++)
        {
            int x = i - center;
            int y = j - center;
            double exponent = -(x * x + y * y) / (2.0 * sigma * sigma);
            matrix[i][j] = constant * exp(exponent);
        }
    }
    return matrix;
}


