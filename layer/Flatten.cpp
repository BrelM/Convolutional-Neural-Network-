#include <iostream>
#include "Flatten.h"

using namespace std;


Flatten::Flatten(){}

vector<double> Flatten::flat(vector<vector<double> > &x)
{

    //cout << "flat" << endl;
    vector<double> vect = x[0];
    for (int i = 1; i < static_cast<int>(x.size()); i++)
        vect.insert(vect.end(), x[i].begin(), x[i].end());

    return vect;
}


