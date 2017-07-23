//
// Created by tri on 22/07/2017.
//

#include "Layer.h"

Layer::Layer(int nNeurals, bool isBias, int activeFunc):nNeurals(nNeurals), isBias(isBias), activeFunc(activeFunc) {
    if(nNeurals >0){
        neurals = mat(nNeurals, 1); // default constructor: mat(,) === Mat<double>(,)
        neurals.zeros(); // first assign for safe
    }
}

void Layer::activation() {

    neurals = activationFunction(neurals, activeFunc);
}

mat Layer::getErrGradient(mat error) {

    mat tmp, gradient;

    switch(activeFunc){
        case SIGMOID: {
            tmp = 1 - neurals;
            gradient = dotProduct(neurals, tmp);
            break;
        }
        case TANH:{
            gradient = 1 - dotProduct(neurals, neurals);
            break;
        }
        case RELU: {

            break;
        }
        default:
            break;
    }

    return dotProduct(gradient, error);
}

void Layer::setNeuralsValue(mat values) {
    neurals = values;
}
