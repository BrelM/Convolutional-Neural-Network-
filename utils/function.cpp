#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>

#include "function.h"

using namespace std;


double min_value(vector<double> v)
{
    double mi(v[0]);
    for(double i:v)
    {
        if(i < mi)
            mi = i;
    }
    return mi;
}

double max_value(vector<double> v)
{
    double ma(v[0]);
    for(double i:v)
    {
        if(i > ma)
            ma = i;
    }
    return ma;
}


vector<vector<double>> init_table(int nblig, int nbcol, double nombre)
{
    vector<vector<double>> table;

    for (int i = 0; i < nblig; ++i)
    {
        std::vector<double> a(nbcol, nombre);
        table.push_back(a);
    }
    return table;
}

vector<vector<double>> paste_image(vector<vector<double>> image, vector<vector<double>> templat, int padding)
{
    int templateLength = templat.size();

    for (int i = padding; i < templateLength - padding; ++i)
    {
        for (int j = padding; i < templateLength - padding; ++j)
        {
            templat[i][j] = image[i - padding][j - padding];
        }
    }
    return templat;
}

vector<vector<double>> add_right_margin(vector<vector<double>> image, int conv_shape, int stride, int xp)
{
    int n = static_cast<int>(image.size());
    double padding = ceil(stride * (xp - 1) + conv_shape - n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < padding; ++j)
            image[i].push_back(0);
    }

    for (int i = 0; i < padding; ++i)
    {
        vector<double> a(n + padding, 0);
        image.push_back(a);
    }

    return image;
}

double get_stride(int x, int sh, int xp)
{
    return ceil((x - sh) / (xp - 1));
}

vector<vector<double>> zscore_norm_matrix(vector<vector<double>>& matrix)
{
    int n = matrix.size();
    double mean = 0.0;
    double std = 0.0;

    // Calcul de la moyenne
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mean += matrix[i][j];
        }
    }
    mean /= (n * n);

    // Calcul de l'écart-type
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std += pow(matrix[i][j] - mean, 2);
        }
    }
    std = sqrt(std / (n * n));

    // Normalisation Z-score
    if (std != 0.0)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[i][j] = (matrix[i][j] - mean) / std;
            }
        }
    }

    return matrix;
}

vector<double> min_max(vector<double>& vec)
{
    int n = vec.size();
    double mi = min_value(vec);
    double ma = max_value(vec);

    // Normalisation Z-score
    if (mi != ma)
    {
        for (int i = 0; i < n; i++)
        {
            vec[i] = (vec[i] - mi) / (ma - mi);
        }
    }

    return vec;
}




vector<double> zscore_norm(vector<double>& vec)
{
    int n = vec.size();
    double mean = 0.0;
    double std = 0.0;

    // Calcul de la moyenne
    for (int i = 0; i < n; i++)
    {
        mean += vec[i];
    }
    mean /= n;

    // Calcul de l'écart-type
    for (int i = 0; i < n; i++)
    {
        std += pow(vec[i] - mean, 2);
    }
    std = sqrt(std / n);

    // Normalisation Z-score
    if (std != 0.0)
    {
        for (int i = 0; i < n; i++)
        {
            vec[i] = (vec[i] - mean) / std;
        }
    }

    return vec;
}

vector<vector<vector<double>>> reshape(vector<double> vect, int nblig, int nbcol)
{

    vector<vector<vector<double>>> mat;

    for (int i = 0; i < nblig; ++i)
    {
        vector<vector<double>> a;
        for (int j = 0; j < nbcol; ++j)
        {
            vector<double> b(nbcol, 0);
            a.push_back(b);
        }
        mat.push_back(a);
    }

    int vectIndex = 0;

    for (int i = 0; i < nblig; ++i)
    {

        for (int j = 0; j < nblig; ++j)
        {

            for (int k = 0; k < nblig; ++k)
            {
                mat[i][j][k] = vect[vectIndex];
                vectIndex++;
            }
        }
    }

    return mat;
}

vector<vector<double>> rotate(vector<vector<double>> mat)
{

    vector<vector<double>> output;

    for (int i = 0; i < static_cast<int>(mat.size()); i++)
    {
        vector<double> b(mat[0].size(), 0);
        output.push_back(b);
    }

    for (int i = 0; i < static_cast<int>(mat.size()); ++i)
    {
        for (int j = 0; j < static_cast<int>(mat[0].size()); ++j)
        {
            output[i - 1][j - 1] = mat[mat.size() - i][mat[0].size() - j];
        }
    }

    return output;
}

vector<vector<double>> diff(vector<vector<double>> mat1, vector<vector<double>> mat2, const double alpha)
{

    if (mat1.size() != mat2.size() or mat1[0].size() != mat2[0].size())
    {
        throw invalid_argument("Matrixes should have the same shape. Instead, they have sizes: " + to_string(mat1.size()) + " and " + to_string(mat2.size()));
    }

    vector<vector<double>> mat;

    for (int i = 0; i < static_cast<int>(mat1.size()); i++)
    {
        vector<double> b(mat1[0].size(), 0);
        mat.push_back(b);
    }

    for (int i = 0; i < static_cast<int>(mat1.size()); ++i)
    {
        for(int j = 0; j < static_cast<int>(mat1[0].size()); ++j){
            mat[i][j] = mat1[i][j] - alpha * mat2[i][j];
        }
    }
    return mat;
}

