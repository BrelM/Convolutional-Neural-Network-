#ifndef POOL_H
#define POOL_H

#include <vector>
#include <cmath>
#include <algorithm>

#include "../utils/function.h"
#include "Pool.h"


using namespace std;

class Pool
{
public:
    int stride;
    int shape;



    Pool();
    Pool(const Pool&);
    Pool(int p_shape, int p_stride);

    ~Pool();

    void init(int shape, int stride);

    vector<vector<vector<double>>> forward(vector<vector<vector<double>>> &x) const;
};

#endif // POOL_H
