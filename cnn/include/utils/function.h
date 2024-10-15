#ifndef FUNCTION_H
#define FUNCTION_H
#include <vector>

using namespace std;



template<typename T>
vector<T> reverse(vector<T>);

template<typename T>
vector<T> reverse(vector<T> list)
{
    vector<T> output;
	while(!list.empty())
	{
		output.push_back(list.back());
		list.pop_back();
	}

	return output;
}


vector<vector<double>> init_table(int nblig, int nbcol, double nombre);
vector<vector<double>> paste_image(vector<vector<double>> image, vector<vector<double>> templat, int padding);
vector<vector<double>> add_right_margin(vector<vector<double>> image, int conv_shape, int stride, int xp);
double get_stride(int x, int sh, int xp);
vector<vector<double>> zscore_norm_matrix(vector<vector<double>>& matrix);

vector<double> min_max(vector<double>& vec);
vector<double> zscore_norm(vector<double>& vec);
vector<vector<vector<double>>> reshape(vector<double> vect, int nblig, int nbcol);
vector<vector<double>> rotate(vector<vector<double>>& mat);
vector<vector<double>> diff(vector<vector<double>> mat1, vector<vector<double>> mat2, const double alpha = 1);

double min_value(vector<double>);
double max_value(vector<double>);


#endif
