// Conv.py

//     Class Conv
//         A class designed to be a convolution layer.Its purpose is to be added as a layer in a neural network standard model.

#include "Conv.h"

using namespace std;


Conv::Conv(): nb_kernel(0), shape(0), stride(0), padding(0), kernel({})
{
}

Conv::Conv(int p_nb_kernel, int p_shape, int p_stride, int p_padding): nb_kernel(p_nb_kernel), shape(p_shape), stride(p_stride), padding(p_padding), kernel({})
{}

Conv::Conv(const Conv& conv): nb_kernel(conv.nb_kernel), shape(conv.shape), stride(conv.stride), padding(conv.padding), kernel(conv.kernel)
{
}


void Conv::init(int p_nb_kernel, int p_shape, int p_stride, int p_padding)
{
    stride = p_stride;
    shape = p_shape;
    nb_kernel = p_nb_kernel;
    padding = p_padding;
    for (int i = 0; i < p_nb_kernel; i++)
        kernel.push_back(gaussianInitKernel(shape));
}


vector<vector<vector<double>>> Conv::forward(const vector<vector<double>> x) const
{
    //cout << "conv0" << endl;
    vector<vector<vector<double>>> y;
    int outWidth = ceil((static_cast<int>(x.size()) - shape + padding) / stride + 1);
    int outHeight = outWidth;
    vector<vector<double>> xCopy = x;

    xCopy = add_right_margin(xCopy, shape, stride, outWidth);

    for (int i = 0; i < nb_kernel; i++)
    {
        vector<vector<double>> a;
        for (int j = 0; j < outHeight; j++)
        {
            vector<double> b(outWidth, 0);
            a.push_back(b);
        }
        y.push_back(a);
    }

    for (int kernelIdx = 0; kernelIdx < static_cast<int>(kernel.size()); kernelIdx++)
    {
        int a = 0;
        for (int i = 0; i < static_cast<int>(xCopy[0].size()) - shape + 1; i += stride)
        {
            int b = 0;
            for (int j = 0; j < static_cast<int>(xCopy.size()) - shape + 1; j += stride)
            {
                double sum_ = 0.0;
                for (int k = i; k < i + shape; k++)
                {
                    for (int l = j; l < j + shape; l++)
                    {
                        sum_ += xCopy[k][l] * kernel[kernelIdx][k - i][l - j];
                    }
                }
                y[kernelIdx][a][b] = ReLU(sum_);
                b++;
            }
            a++;
        }
    }

    //cout << "conv0_out" << endl;
    return y;
}



vector<vector<vector<double>>> Conv::forward(const vector<vector<vector<double>>> x) const
{

    //cout << "conv1" << endl;
    vector<vector<vector<double>>> y;

    int outWidth = ceil((x[0][0].size() - shape + padding) / static_cast<double>(stride) + 1);
    vector<vector<vector<double>>> xCopy = x;
    for (auto &matrix : xCopy)
        matrix = add_right_margin(matrix, shape, stride, outWidth);

    int outHeight = outWidth;

    for (int i = 0; i < nb_kernel; ++i)
    {
        vector<vector<double>> a;
        for (int j = 0; j < outHeight; ++j)
        {
            vector<double> b(outWidth, 0);
            a.push_back(b);
        }
        y.push_back(a);
    }

    for (int kernelIdx = 0; kernelIdx < static_cast<int>(kernel.size()); kernelIdx++)
    {
        int a = 0;
        for (int i = 0; i < static_cast<int>(xCopy[0][0].size()) - shape + 1; i += stride)
        {
            int b = 0;
            for (int j = 0; j < static_cast<int>(xCopy[0].size()) - shape + 1; j += stride)
            {
                double sum_ = 0.0;
                for (int k = i; k < i + shape; k++)
                {
                    for (int l = j; l < j + shape; l++)
                    {
                        sum_ += xCopy[kernelIdx][k][l] * kernel[kernelIdx][k - i][l - j];
                    }
                }
                y[kernelIdx][a][b] = ReLU(sum_);
                b++;
            }
            a++;
        }
    }

    //cout << "conv1_out" << y[0].size() << endl;
    return y;

}




vector<vector<vector<double>>> Conv::update(const vector<vector<vector<double>>> &gradients,
                                            vector<vector<vector<double>>> &inputs0,
                                            vector<vector<vector<double>>> &inputs1,
                                            double alpha)
{
    vector<vector<vector<double>>> y;
    //if (is_same<decltype(inputs0[0][0]), vector<double>>::value)

    int stride = get_stride(inputs0[0].size(), gradients[0][0].size(), this->shape);
    for (int i = 0; i < nb_kernel; i++)
        inputs0[i] = add_right_margin(inputs0[i], gradients[0][0].size(), stride, this->shape);

    int outWidth = ceil((inputs0[0][0].size() - gradients[0][0].size() + this->padding) / static_cast<double>(stride) + 1);
    int outHeight = outWidth;


    for (int i = 0; i < nb_kernel; ++i)
    {
        vector<vector<double>> a;
        for (int j = 0; j < outHeight; ++j)
        {
            vector<double> b(outWidth, 0);
            a.push_back(b);
        }
        y.push_back(a);
    }

    for (int kernelIdx = 0; kernelIdx < nb_kernel; kernelIdx++)
    {
        int a = 0;
        for (int i = 0; i < static_cast<int>(inputs0[kernelIdx].size()) - static_cast<int>(gradients[0][0].size()) + 1; i += stride)
        {
            int b = 0;
            for (int j = 0; j < static_cast<int>(inputs0[kernelIdx][0].size()) - static_cast<int>(gradients[0][0].size()) + 1; j += stride)
            {
                double sum_ = 0.0;
                for (int k = i; k < i + static_cast<int>(gradients[0][0].size()); k++)
                {
                    for (int l = j; l < j + static_cast<int>(gradients[0][0].size()); l++)
                    {
                        sum_ += inputs0[kernelIdx][k][l] * gradients[kernelIdx][k - i][l - j];
                    }
                }
                y[kernelIdx][a][b] = ReLU(sum_);
                b++;
            }
            a++;
        }
    }

    for (int kernelIdx = 0; kernelIdx < nb_kernel; kernelIdx++)
    {
        kernel[kernelIdx] = diff(kernel[kernelIdx], y[kernelIdx], alpha);
        kernel[kernelIdx] = zscore_norm_matrix(kernel[kernelIdx]);
    }

    //if (is_same<decltype(inputs1[0][0]), vector<double>>::value)
    for (int i = 0; i < nb_kernel; i++)
    {
        inputs1[i] = add_right_margin(inputs1[i], shape, stride, inputs0[0][0].size());
    }

    outWidth = ceil((inputs1[0][0].size() - shape + padding) / static_cast<double>(stride) + 1);
    outHeight = outWidth;

    y.clear();

    for (int i = 0; i < nb_kernel; ++i)
    {
        vector<vector<double>> a;
        for (int j = 0; j < outHeight; ++j)
        {
            vector<double> b(outWidth, 0);
            a.push_back(b);
        }
        y.push_back(a);
    }


    for (int kernelIdx = 0; kernelIdx < nb_kernel; kernelIdx++)
    {
        int a = 0;
        for (int i = 0; i < static_cast<int>(inputs1[kernelIdx].size()) - shape + 1; i += stride)
        {
            int b = 0;
            for (int j = 0; j < static_cast<int>(inputs1[kernelIdx].size()) - shape + 1; j += stride)
            {
                double sum_ = 0.0;
                for (int k = i; k < i + shape; k++)
                {
                    for (int l = j; l < j + shape; l++)
                    {
                        sum_ += inputs1[kernelIdx][k][l] * kernel[kernelIdx][k - i][l - j];
                    }
                }
                y[kernelIdx][a][b] = ReLU(sum_);
                b++;
            }
            a++;
        }
    }

    return y;
}




vector<vector<vector<double>>> Conv::update(const vector<vector<vector<double>>> &gradients,
                                            vector<vector<vector<double>>> &inputs0,
                                            vector<vector<double>> &inputs1,
                                            double alpha)
{
    vector<vector<vector<double>>> y;
    //if (is_same<decltype(inputs0[0][0]), vector<double>>::value)

    int stride = get_stride(inputs0[0].size(), gradients[0][0].size(), shape);
    for (int i = 0; i < nb_kernel; i++)
        inputs0[i] = add_right_margin(inputs0[i], gradients[0][0].size(), stride, shape);


    int outWidth = ceil((inputs0[0][0].size() - gradients[0][0].size() + padding) / static_cast<double>(stride) + 1);
    int outHeight = outWidth;

    for (int i = 0; i < nb_kernel; ++i)
    {
        vector<vector<double>> a;
        for (int j = 0; j < outHeight; ++j)
        {
            vector<double> b(outWidth, 0);
            a.push_back(b);
        }
        y.push_back(a);
    }

    for (int kernelIdx = 0; kernelIdx < nb_kernel; kernelIdx++)
    {
        int a = 0;
        for (int i = 0; i < static_cast<int>(inputs0[kernelIdx].size()) - static_cast<int>(gradients[0][0].size()) + 1; i += stride)
        {
            int b = 0;
            for (int j = 0; j < static_cast<int>(inputs0[kernelIdx][0].size()) - static_cast<int>(gradients[0][0].size()) + 1; j += stride)
            {
                double sum_ = 0.0;
                for (int k = i; k < i + static_cast<int>(gradients[0][0].size()); k++)
                {
                    for (int l = j; l < j + static_cast<int>(gradients[0][0].size()); l++)
                    {
                        sum_ += inputs0[kernelIdx][k][l] * gradients[kernelIdx][k - i][l - j];
                    }
                }
                y[kernelIdx][a][b] = ReLU(sum_);
                b++;
            }
            a++;
        }
    }

    for (int kernelIdx = 0; kernelIdx < nb_kernel; kernelIdx++)
    {
        kernel[kernelIdx] = diff(kernel[kernelIdx], y[kernelIdx], alpha);
        kernel[kernelIdx] = zscore_norm_matrix(kernel[kernelIdx]);
    }

    //if (!is_same<decltype(inputs1[0][0]), vector<double>>::value)
    inputs1 = add_right_margin(inputs1, shape, this->stride, inputs0[0][0].size());
    outWidth = ceil((inputs1[0].size() - shape + padding) / static_cast<double>(stride) + 1);
    outHeight = outWidth;
    y.clear();

    for (int i = 0; i < nb_kernel; ++i)
    {
        vector<vector<double>> a;
        for (int j = 0; j < outHeight; ++j)
        {
            vector<double> b(outWidth, 0);
            a.push_back(b);
        }
        y.push_back(a);
    }

    for (int kernelIdx = 0; kernelIdx < nb_kernel; kernelIdx++)
    {
        int a = 0;
        for (int i = 0; i < static_cast<int>(inputs1.size()) - shape + 1; i += stride)
        {
            int b = 0;
            for (int j = 0; j < static_cast<int>(inputs1.size()) - shape + 1; j += stride)
            {
                double sum_ = 0.0;
                for (int k = i; k < i + shape; k++)
                {
                    for (int l = j; l < j + shape; l++)
                    {
                        sum_ += inputs1[k][l] * kernel[kernelIdx][k - i][l - j];
                    }
                }
                y[kernelIdx][a][b] = ReLU(sum_);
                b++;
            }
            a++;
        }
    }

    return y;
}



void Conv::update(const vector<vector<vector<double>>> &gradients,
                                                        vector<vector<double>> &inputs0,
                                                        double alpha)
{
    vector<vector<vector<double>>> y;

    int stride = get_stride(inputs0.size(), gradients[0][0].size(), this->shape);
    inputs0 = add_right_margin(inputs0, gradients[0][0].size(), stride, this->shape);
    int outWidth = ceil((inputs0.size() - gradients[0][0].size() + this->padding) / static_cast<double>(stride) + 1);
    int outHeight = outWidth;

    for (int i = 0; i < nb_kernel; ++i)
    {
        vector<vector<double>> a;
        for (int j = 0; j < outHeight; ++j)
        {
            vector<double> b(outWidth, 0);
            a.push_back(b);
        }
        y.push_back(a);
    }

    for (int kernelIdx = 0; kernelIdx < nb_kernel; kernelIdx++)
    {
        int a = 0;
        for (int i = 0; i < static_cast<int>(inputs0.size()) - static_cast<int>(gradients[kernelIdx].size()) + 1; i += stride)
        {
            int b = 0;
            for (int j = 0; j < static_cast<int>(inputs0.size()) - static_cast<int>(gradients[kernelIdx].size()) + 1; j += stride)
            {
                double sum_ = 0.0;
                for (int k = i; k < i + static_cast<int>(gradients[kernelIdx].size()); k++)
                {
                    for (int l = j; l < j + static_cast<int>(gradients[kernelIdx].size()); l++)
                    {
                        sum_ += inputs0[k][l] * gradients[kernelIdx][k - i][l - j];
                    }
                }
                y[kernelIdx][a][b] = ReLU(sum_);
                b++;
            }
            a++;
        }
    }

    for (int kernelIdx = 0; kernelIdx < nb_kernel; kernelIdx++)
    {
        kernel[kernelIdx] = diff(kernel[kernelIdx], y[kernelIdx], alpha);
        kernel[kernelIdx] = zscore_norm_matrix(kernel[kernelIdx]);
    }

}



Conv::~Conv(){}


