//
// Created by tri on 13/07/2017.
//
#include "BatchTrainer.h"

BatchTrainer::BatchTrainer( NeuralNetwork* nn, int bSize ):Trainer(nn),
                                                                         NN(nn), epoch(0),
                                                                         learningRate(LEARNING_RATE),
                                                                         desiredAccuracy(DESIRED_ACCURACY),
                                                                         maxEpochs(MAX_EPOCHS),
                                                                         trainingSetAccuracy(0),
                                                                         validationSetAccuracy(0),
                                                                         generalizationSetAccuracy(0),
                                                                         validationSetMSE(0),
                                                                         generalizationSetMSE(0),
                                                                         batchSize(bSize) {
    int nLayers = nn->nLayer;

    for(int i=0; i< nLayers - 1; i++){
        int layerSize = nn->layers[i]->nNeurals;
        int nextLayerSize = nn->layers[i+1]->nNeurals;

        mat deltaW = mat((const uword) nextLayerSize, (const uword) layerSize);
        deltaW.zeros();
        sumDeltaWeights.push_back(deltaW);

        mat deltaB;
        if(nn->layers[i]->isBias){
            deltaB = mat((const uword) nextLayerSize, 1);
            deltaB.zeros();
        }

        // if this layer has no bias => push empty(size = [0x0]); and Output layer has no bias.
        sumDeltaBiass.push_back(deltaB);
    }
}

void BatchTrainer::resetSumDelta() {

    for(int i =0; i< NN->nLayer -1; i++){
        sumDeltaWeights[i].zeros();
        sumDeltaBiass[i].zeros();
    }

}

void BatchTrainer::updateWeights() {

    int nLayers = NN->nLayer;
    vector<mat> deltaWs((unsigned long) (nLayers - 1));
    vector<mat> deltaBs((unsigned long) (nLayers - 1));

    for(int i =0; i< nLayers - 1; i++){
        deltaWs[i] = 1.0/batchSize * sumDeltaWeights[i] + LAMDA * NN->weights[i];
        deltaBs[i] = 1.0/batchSize * sumDeltaBiass[i];
    }
    NN->updateWeights(deltaWs, deltaBs, learningRate);
}

void BatchTrainer::backpropagate(mat desiredOutputs, int index) {

    //modify deltas between layers
    int nLayers = NN->nLayer;

    mat err = desiredOutputs - NN->layers[nLayers-1]->neurals;

    for(int i=nLayers-1; i >=1; i--){

        mat errorGradient = NN->layers[i]->getErrGradient(err);
        mat deltaW = errorGradient * NN->layers[i-1]->neurals.t();
        sumDeltaWeights[i-1] += deltaW;
        mat deltaB;

        if( NN->biass[i-1].n_rows != 0 ){
            deltaB = errorGradient;
            sumDeltaBiass[i-1] += deltaB;
        }
        err = NN->weights[i-1] .t() * errorGradient;
    }

    if( (index +1) % batchSize ==0 ){
        updateWeights();
        resetSumDelta();
    }
}

void BatchTrainer::runTrainingEpoch(std::vector<DataEntry *> trainingSet) {
    //incorrect patterns
    double incorrectPatterns = 0; // use in classification
    double mse = 0; //use in regression

    int size = (int)trainingSet.size();
    //for every training pattern
    for ( int tp = 0; tp < size; tp++) {

        //feed inputs through network and backpropagate errors
        NN->feedForwardPattern( trainingSet[tp]->pattern );
        backpropagate( trainingSet[tp]->target, tp );

        switch (NN->netType){
            case CLASSIFICATION: {
                //pattern correct flag
                bool patternCorrect = true;
                //check all outputs from neural network against desired values
                //pattern incorrect if desired and output differ
                patternCorrect = checkOutput(NN->clampOutput(), trainingSet[tp]->target);
                //if pattern is incorrect add to incorrect count
                if (!patternCorrect) incorrectPatterns++;
                break;
            }
            case parameters::REGRESSTION: {
                //increase mean square error
                mse += (NN->getOutput()(0,0) - trainingSet[tp]->target(0, 0)) *
                       (NN->getOutput()(0,0) - trainingSet[tp]->target(0, 0));
                break;
            }
        }
    }//end for

    //update training accuracy and MSE
    switch (NN->netType){
        case CLASSIFICATION: {
            trainingSetAccuracy = 100 - (incorrectPatterns / trainingSet.size() * 100);
            break;
        }
        case REGRESSTION: {
            //increase mean square error
            this->trainingSetMSE = mse / trainingSet.size();
            break;
        }
    }
}