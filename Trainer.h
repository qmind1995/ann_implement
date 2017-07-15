//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_TRAINER_H
#define ANN_IMPLEMENT_TRAINER_H

#include <fstream>
#include <vector>

//neural network header
#include "NeuralNetwork.h"

//Constant Defaults!
#define LEARNING_RATE 0.01

#define MOMENTUM 0.9
#define MAX_EPOCHS 100
#define DESIRED_ACCURACY 95
#define DESIRED_MSE 0.001

using namespace arma;

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
    mat deltaInputHidden;
    mat deltaHiddenOutput;

    //error gradients
    mat hiddenErrorGradients;
    mat outputErrorGradients;

    //accuracy stats per epoch
    double trainingSetAccuracy;
    double validationSetAccuracy;
    double generalizationSetAccuracy;
    double trainingSetMSE;
    double validationSetMSE;
    double generalizationSetMSE;

    //batch learning flag

    //log file handle

public:
    Trainer();
    Trainer( NeuralNetwork* untrainedNetwork );
    void trainNetwork( trainingDataSet* tSet );


    inline mat getOutputErrorGradient( mat desiredValue, mat outputValue );
    inline mat getHiddenErrorGradient(mat outErrGradients);
    bool checkOutput(mat output, mat target);
private:

    mat dotProduct(mat A, mat B);
    void runTrainingEpoch( std::vector<DataEntry*> trainingSet );
    void backpropagate(mat desiredOutputs);
    void updateWeights();

};


#endif //ANN_IMPLEMENT_TRAINER_H
