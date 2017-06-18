//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_NEURALNETWORK_H
#define ANN_IMPLEMENT_NEURALNETWORK_H

#include "DataReader.h"

class NeuralNetwork{
private:

    //number of neurons
    int nInput, nHidden, nOutput;

    //neurons
    Mat inputNeurons;
    Mat hiddenNeurons;
    Mat outputNeurons;

    //weights
    Mat wInputHidden;
    Mat wHiddenOutput;

    friend neuralNetworkTrainer;

public:

    //constructor & destructor
    NeuralNetwork(int numInput, int numHidden, int numOutput);
    ~NeuralNetwork();

    //weight operations
//    bool loadWeights(char* inputFilename);
//    bool saveWeights(char* outputFilename);
    int* feedForwardPattern( double* pattern );
//    double getSetAccuracy( std::vector<dataEntry*>& set );
//    double getSetMSE( std::vector<dataEntry*>& set );

private:

    void initializeWeights();
    inline double activationFunction( double x );
    inline int clampOutput( double x );
    void feedForward( double* pattern );
};
#endif //ANN_IMPLEMENT_NEURALNETWORK_H