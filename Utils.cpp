//
// Created by tri on 22/07/2017.
//
#ifndef UTILS_CPP
#define UTILS_CPP
#include "armadillo"


class NeuralNetwork;

namespace parameters{
    static const int NONE = 0;
    static const int SIGMOID = 1;
    static const int TANH = 2;
    static const int RELU = 3;
    static const int CLASSIFICATION = 10;
    static const int REGRESSTION = 11;
    static const double PI = 3.141592654;
    struct paramHolder {
        int argc;
        char ** argv;
        NeuralNetwork* net;
    };

}

using namespace std;
using namespace arma;
using namespace parameters;

inline static mat activationFunction(mat neurals, int activeFunc){

    int nNeurals = neurals.n_rows;

    for(int i=0 ;i< nNeurals; i++) {

        double x = neurals(i, 0);

        switch (activeFunc){
            case SIGMOID: {
                neurals(i, 0) = 1 / (1 + exp(-x));
                break;
            }
            case TANH: {
                neurals(i, 0) = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
                break;
            }
            case RELU: {

                break;
            }
            default:
                break;
        }
    }
    return neurals;
}

inline static mat dotProduct(mat A, mat B){

    auto sizeA = arma::size(A);
    auto sizeB = arma::size(B);
    if(sizeA.n_rows != sizeB.n_rows || sizeA.n_cols != sizeB.n_cols){
        cout<<"dont match size A and B ! check again idot!"<<endl;
    }
    mat output = mat(sizeA.n_rows, sizeA.n_cols);
    for(int i =0; i< sizeA.n_rows; i++){
        for(int j=0; j< sizeA.n_cols; j++){
            output(i,j) = A(i,j) * B(i,j);
        }
    }
    return output;
}


inline static double uniformRandom(double floor, double ceil){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(floor, ceil);
    return dis(gen);
}

inline static double gaussianRamdom(double mean, double neighbour){
    std::random_device rd;
    std::mt19937 gen(rd());

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<> d(mean, neighbour);
    return d(gen);
}

inline static string activeFuncNameToString(int function){
    switch(function){
        case TANH: {
            return "Tanh";
        }
        case SIGMOID: {
            return "Sigmoid";
        }
        case RELU: {
            return "Relu";
        }
        case parameters::NONE: {
            return "None";
        }
        default:
            return "Unknown";
    }
}

#endif