//
// Created by tri on 22/07/2017.
//

#ifndef FAKE_CODE_LAYER_H
#define FAKE_CODE_LAYER_H

#include "Utils.cpp"

class Trainer;
class NeuralNetwork;

namespace constant{
    static const int NONE = 0;
    static const int SIGMOID = 1;
    static const int TANH = 2;
    static const int RELU = 3;
}

using namespace arma;
using namespace constant;

class Layer{

public:
    Layer(int nNeurals, bool isBias, int activeFunc = constant::NONE);
    mat getGradient(mat error);
    int nNeurals;
    bool isBias;
    void setNeuralsValue(mat values);
    void activation();
    friend NeuralNetwork;
    friend Trainer;
private:
    int activeFunc;

    mat neurals;
};
#endif //FAKE_CODE_LAYER_H
