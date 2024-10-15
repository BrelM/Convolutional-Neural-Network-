

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>
#include <utility>
#include <iostream>


class Layer
{
    public:
        std::string name;

        Layer(std::string p_name);
        virtual ~Layer();
        //virtual void init(auto& a, ...) = 0;
        //virtual void forward(auto& a, ...) = 0;
        //virtual void update(auto& a, ...) = 0;

};



#endif //LAYER_H