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
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 90
#define DESIRED_MSE 0.001

class NeuralNetworkTrainer
{

private:

    //network to be trained
    NeuralNetwork* NN;

    //learning parameters
    double learningRate;					// adjusts the step size of the weight update
    double momentum;						// improves performance of stochastic learning (don't use for batch)

    //epoch counter
    long epoch;
    long maxEpochs;

    //accuracy/MSE required
    double desiredAccuracy;

    //change to weights
    double** deltaInputHidden;
    double** deltaHiddenOutput;

    //error gradients
    double* hiddenErrorGradients;
    double* outputErrorGradients;

    //accuracy stats per epoch
    double trainingSetAccuracy;
    double validationSetAccuracy;
    double generalizationSetAccuracy;
    double trainingSetMSE;
    double validationSetMSE;
    double generalizationSetMSE;

    //batch learning flag
    bool useBatch;

    //log file handle
    bool loggingEnabled;
    std::fstream logFile;
    int logResolution;
    int lastEpochLogged;

public:

    neuralNetworkTrainer( NeuralNetwork* untrainedNetwork );
    void setTrainingParameters( double lR, double m, bool batch );
    void setStoppingConditions( int mEpochs, double dAccuracy);
    void useBatchLearning( bool flag ){ useBatch = flag; }
    void enableLogging( const char* filename, int resolution );

    void trainNetwork( trainingDataSet* tSet );

private:
    inline double getOutputErrorGradient( double desiredValue, double outputValue );
    double getHiddenErrorGradient( int j );
    void runTrainingEpoch( std::vector<dataEntry*> trainingSet );
    void backpropagate(double* desiredOutputs);
    void updateWeights();
};


#endif //ANN_IMPLEMENT_TRAINER_H
