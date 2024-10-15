#ifndef FLATTEN_H
#define FLATTEN_H

#include <vector>

class Flatten
{
public:
    Flatten();

    std::vector<double> flat(std::vector<std::vector<double> > &x);
};

#endif // FLATTEN_H
