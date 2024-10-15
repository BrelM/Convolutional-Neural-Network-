#include <iostream>
#include "Pool.h"

Pool::Pool(int p_shape, int p_stride) : stride(p_stride), shape(p_shape) {}


Pool::Pool(const Pool& pool): stride(pool.stride), shape(pool.shape){}


Pool::Pool(): stride(1), shape(3)
{}


void Pool::init(int p_shape, int p_stride)
{
    shape = p_shape;
    stride = p_stride;
}


vector<vector<vector<double>>> Pool::forward(vector<vector<vector<double>>> &x)const
{
    vector<vector<vector<double>>> y;
    //cout << "pool" << endl;
    int outHeight = static_cast<int>((x[0].size() - shape) / stride + 1);
    int outWidth = ceil((x[0][0].size() - shape) / stride + 1);

    for (auto img : x)
        img = add_right_margin(img, shape, stride, outWidth);

    for (int i = 0; i < static_cast<int>(x.size()); ++i)
    {
        vector<vector<double>> a;
        for (int j = 0; j < outHeight; ++j)
        {
            vector<double> b(outWidth, 0);
            a.push_back(b);
        }
        y.push_back(a);
    }


    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
        for (int j = 0; j < outHeight - shape + 1; j += stride)
        {
            for (int k = 0; k < outWidth - shape + 1; k += stride)
            {
                double max_ = x[i][j][k];
                for (int l = j; l < j + shape; l++)
                {
                    for (int m = k; m < k + shape; m++)
                    {
                        max_ = max(max_, x[i][l][m]);
                    }
                }
                y[i][j][k] = max_;
            }
        }
    }

    //cout << "pool_out" << y[0].size() << endl;

    return y;
}




Pool::~Pool(){}






