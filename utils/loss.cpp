
#include "loss.h"

double MSE(std::vector<double> out)
{
    double sum = 0.0;
    for (double value : out)
    {
        sum += value * value;
    }
    return sum / out.size();
}