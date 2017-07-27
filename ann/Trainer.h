//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_TRAINER_H
#define ANN_IMPLEMENT_TRAINER_H

#include <fstream>
#include <vector>

//neural network header
#include "./NeuralNetwork.h"

//Constant Defaults!
#define LEARNING_RATE 0.1

#define MOMENTUM 0.9
#define MAX_EPOCHS 1000
#define DESIRED_ACCURACY 95
#define DESIRED_MSE 0.001

using namespace arma;
using namespace parameters;

class Trainer {

private:

    //network to be trained
    NeuralNetwork* NN;

    //learning parameters
    double learningRate;					// adjusts the step size of the weight update

    //epoch counter
    long epoch;
    long maxEpochs;

    //accuracy/MSE required
    double desiredAccuracy;

    //change to weights
    vector<mat> deltaWeights;
    vector<mat> deltaBiass;

    //error gradients
    vector<mat> errorGradients;

    //accuracy stats per epoch
    double trainingSetAccuracy;
    double validationSetAccuracy;
    double generalizationSetAccuracy;

    double validationSetMSE;
    double generalizationSetMSE;

    //batch learning flag

    //log file handle

public:
    double trainingSetMSE;

    Trainer();
    Trainer( NeuralNetwork* untrainedNetwork );
    void trainNetwork(trainingDataSet* tSet );
    bool checkOutput(mat output, mat target);
private:
    virtual void runTrainingEpoch( std::vector<DataEntry*> trainingSet );
    void backpropagate(mat desiredOutputs);
    inline void updateWeights();

};


#endif //ANN_IMPLEMENT_TRAINER_H
