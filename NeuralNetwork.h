//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_NEURALNETWORK_H
#define ANN_IMPLEMENT_NEURALNETWORK_H

#include "DataReader.h"
#include "Layer.h"

using namespace arma;
using namespace constant;

class Trainer;
class BatchTrainer;

class NeuralNetwork{

public:

    int netType;

    //constructor & destructor
    NeuralNetwork(vector<Layer*> nlayers, int type);
    ~NeuralNetwork();

    //weight operations
    NeuralNetwork(string weightFileName);
    bool saveWeights(string outputFilename);
    mat feedForwardPattern(arma::mat input);
    double getSetAccuracy( std::vector<DataEntry*>& set );
    arma::mat clampOutput();
    void updateWeights(vector<mat> deltaWeights, vector<mat> deltaBiass, double learningRate);
    mat getOutput();
    void printNetwokInfo();
private:

    int nLayer;
    //neurons
    vector<Layer*> layers;
    //weights
    vector<mat> weights;
    //bias
    vector<mat> biass;

    friend Trainer;
    friend BatchTrainer;

    bool checkOutput(mat output, mat target);
    mat initializeWeights(int nRows, int nCols);
};
#endif //ANN_IMPLEMENT_NEURALNETWORK_H