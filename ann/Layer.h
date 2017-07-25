//
// Created by tri on 22/07/2017.
//

#ifndef FAKE_CODE_LAYER_H
#define FAKE_CODE_LAYER_H

#include "../Utils.cpp"

class Trainer;
class NeuralNetwork;
class BatchTrainer;

using namespace arma;
using namespace parameters;

class Layer{

public:
    Layer(int nNeurals, bool isBias, int activeFunc = parameters::NONE);
    mat getErrGradient(mat error);
    int nNeurals;
    bool isBias;
    void setNeuralsValue(mat values);
    void activation();
    friend NeuralNetwork;
    friend Trainer;
    friend BatchTrainer;
private:
    int activeFunc;

    mat neurals;
};
#endif //FAKE_CODE_LAYER_H
