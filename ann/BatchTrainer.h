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

    NeuralNetwork* NN;

    vector<mat> sumDeltaWeights;
    vector<mat> sumDeltaBiass;
    double trainingSetAccuracy;
    double validationSetAccuracy;
    double generalizationSetAccuracy;
//    double trainingSetMSE;
    double validationSetMSE;
    double generalizationSetMSE;
public:
    BatchTrainer( NeuralNetwork* untrainedNetwork, int batchSize );
private:
    void backpropagate( mat desiredOutputs,int index );
    void resetSumDelta();
    void runTrainingEpoch( std::vector<DataEntry*> trainingSet );
    inline void updateWeights();
};

#endif //FAKE_CODE_BATCHTRAINER_H
