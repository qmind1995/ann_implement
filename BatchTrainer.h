//
// Created by tri on 13/07/2017.
//

#ifndef FAKE_CODE_BATCHTRAINER_H
#define FAKE_CODE_BATCHTRAINER_H

//neural network header
#include "NeuralNetwork.h"
#include "Trainer.h"

#define LAMDA 0

using namespace arma;

class BatchTrainer: public Trainer{
private:
    int batchSize;
    //learning parameters
    double learningRate;					// adjusts the step size of the weight update

    //epoch counter
    long epoch;
    long maxEpochs;

    //accuracy/MSE required
    double desiredAccuracy;

    //change to weights
    mat deltaW_InputHidden;
    mat deltaW_HiddenOutput;
    mat deltaB_InputHidden;
    mat deltaB_HiddenOutput;

    //error gradients
    mat hiddenErrorGradients;
    mat outputErrorGradients;
    NeuralNetwork* NN;

    mat sumDeltaW_InputHidden;
    mat sumDeltaW_HiddenOutput;
    mat sumDeltaB_InputHidden;
    mat sumDeltaB_HiddenOutput;

    double trainingSetAccuracy;
    double validationSetAccuracy;
    double generalizationSetAccuracy;
    double trainingSetMSE;
    double validationSetMSE;
    double generalizationSetMSE;
public:
    BatchTrainer( NeuralNetwork* untrainedNetwork, int batchSize );
    void trainNetwork( trainingDataSet* tSet );
private:
    void updateWeights();
    void backpropagate( mat desiredOutputs,int index );
    void runTrainingEpoch( std::vector<DataEntry*> trainingSet );
};

#endif //FAKE_CODE_BATCHTRAINER_H
