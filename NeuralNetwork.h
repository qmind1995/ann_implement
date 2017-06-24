//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_NEURALNETWORK_H
#define ANN_IMPLEMENT_NEURALNETWORK_H

#include "DataReader.h"
class Trainer;
//class DataReader;
class NeuralNetwork{
private:

    //number of neurons
    int nInput, nHidden, nOutput;

    //neurons
    arma::mat inputNeurons;
    arma::mat hiddenNeurons;
    arma::mat outputNeurons;

    //weights
    arma::mat wInputHidden;
    arma::mat wHiddenOutput;

    friend Trainer;

public:

    //constructor & destructor
    NeuralNetwork(int numInput, int numHidden, int numOutput);
    ~NeuralNetwork();

    //weight operations
//    bool loadWeights(char* inputFilename);
//    bool saveWeights(char* outputFilename);
    arma::mat feedForwardPattern(arma::mat input);
    double getSetAccuracy( std::vector<DataEntry*>& set );
//    double getSetMSE( std::vector<dataEntry*>& set );
    arma::mat clampOutput();

private:
    bool checkOutput(arma::mat output, arma::mat target);
    void initializeWeights();
    inline double activationFunction( double x );
//    inline arma::mat clampOutput();
    void feedForward( arma::mat input );
};
#endif //ANN_IMPLEMENT_NEURALNETWORK_H