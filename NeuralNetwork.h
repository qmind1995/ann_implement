//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_NEURALNETWORK_H
#define ANN_IMPLEMENT_NEURALNETWORK_H

#include "DataReader.h"
#include "Layer.h"

using namespace arma;

class Trainer;
class BatchTrainer;

class NeuralNetwork{
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
public:

    //constructor & destructor
    NeuralNetwork(vector<Layer*> nlayers);
    ~NeuralNetwork();

    //weight operations
    NeuralNetwork(string weightFileName);
    bool saveWeights(string outputFilename);
    mat feedForwardPattern(arma::mat input);
    double getSetAccuracy( std::vector<DataEntry*>& set );
    arma::mat clampOutput();
    void updateWeights(vector<mat> deltaWeights, vector<mat> deltaBiass);

private:
    bool checkOutput(mat output, mat target);
    mat initializeWeights(int nRows, int nCols);
    void feedForward( arma::mat input );
};
#endif //ANN_IMPLEMENT_NEURALNETWORK_H