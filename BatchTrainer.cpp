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
                                                                         trainingSetMSE(0),
                                                                         validationSetMSE(0),
                                                                         generalizationSetMSE(0),
                                                                         batchSize(bSize) {

    deltaW_InputHidden = mat(nn->nHidden, nn->nInput);
    deltaW_InputHidden.zeros();

    deltaW_HiddenOutput = mat(nn->nOutput, nn->nHidden);
    deltaW_HiddenOutput.zeros();

    deltaB_InputHidden = mat(nn->nHidden, 1);
    deltaB_InputHidden.zeros();
    deltaB_HiddenOutput = mat(nn->nOutput, 1);
    deltaB_HiddenOutput.zeros();
    //create error gradient storage
    //--------------------------------------------------------------------------------------------------------
    hiddenErrorGradients = mat(nn->nHidden ,1);
    hiddenErrorGradients.zeros();

    outputErrorGradients = mat(nn->nOutput ,1);
    outputErrorGradients.zeros();

    sumDeltaW_InputHidden = mat(nn->nHidden, nn->nInput);
    sumDeltaW_InputHidden.zeros();

    sumDeltaW_HiddenOutput = mat(nn->nOutput, nn->nHidden);
    sumDeltaW_HiddenOutput.zeros();

    sumDeltaB_InputHidden = mat(nn->nHidden, 1);
    sumDeltaB_InputHidden.zeros();

    sumDeltaB_HiddenOutput = mat(nn->nOutput, 1);
    sumDeltaB_HiddenOutput.zeros();
}

void BatchTrainer::backpropagate(mat desiredOutputs, int index) {
    outputErrorGradients = Trainer::getOutputErrorGradient(desiredOutputs, NN->outputNeurons);
    mat hiddenNoBias(NN->nHidden,1);
    for(int i=0; i<NN->nHidden; i++){
        hiddenNoBias(i,0) = NN->hiddenNeurons(i,0);
    }
    //deltaW_HiddenOutput += outputErrorGradients * hiddenNoBias.t();  // no bias
    sumDeltaW_HiddenOutput += outputErrorGradients * hiddenNoBias.t();
    //deltaB_HiddenOutput += outputErrorGradients;
    sumDeltaB_HiddenOutput += outputErrorGradients;

    hiddenErrorGradients = this->Trainer::getHiddenErrorGradient(outputErrorGradients);
    mat inputNoBias(NN->nInput,1);
    for(int i=0; i<NN->nInput; i++){
        inputNoBias(i,0) = NN->inputNeurons(i,0);
    }

    sumDeltaW_InputHidden += hiddenErrorGradients * inputNoBias.t();
    sumDeltaB_InputHidden += hiddenErrorGradients;
    if( (index +1) % batchSize ==0 ){

        mat wIH_noBias(NN->nHidden, NN->nInput);
        for(int i =0; i<NN->nHidden; i++){
            for(int  j = 0; j< NN->nInput; j++){
                wIH_noBias(i,j) = NN->wInputHidden(i,j);
            }
        }

        mat wHO_noBias(NN->nOutput, NN->nHidden);
        for(int i =0; i<NN->nOutput; i++){
            for(int  j = 0; j< NN->nHidden; j++){
                wHO_noBias(i,j) = NN->wHiddenOutput(i,j);
            }
        }
        deltaW_InputHidden = 1.0/batchSize * sumDeltaW_InputHidden + LAMDA *wIH_noBias;
        deltaB_InputHidden = 1.0/batchSize * sumDeltaB_InputHidden;

        deltaW_HiddenOutput = 1.0/batchSize * sumDeltaW_HiddenOutput + LAMDA *wHO_noBias;
        deltaB_HiddenOutput = 1.0/batchSize * sumDeltaB_HiddenOutput;

        updateWeights();
        sumDeltaB_InputHidden.zeros();
        sumDeltaB_HiddenOutput.zeros();
        sumDeltaW_HiddenOutput.zeros();
        sumDeltaW_InputHidden.zeros();
    }
}

void BatchTrainer::updateWeights() {
    mat deltaInputHidden = mat(NN->nHidden, NN->nInput + 1);
    mat deltaHiddenOutput = mat(NN->nOutput, NN->nHidden +1);

    for(int i =0; i< NN->nHidden; i++){
        for(int j = 0; j < NN->nInput +1 ;j++){
            if(j < NN->nInput){
                deltaInputHidden(i,j) = deltaW_InputHidden(i,j);
            }
            else{
                deltaInputHidden(i,j) = deltaB_InputHidden(i,0);
            }

        }
    }

//    cout<<deltaInputHidden<<endl;

    for(int i =0; i< NN->nOutput; i++) {
        for (int j = 0; j < NN->nHidden + 1; j++) {
            if(j < NN->nHidden){
                deltaHiddenOutput(i,j) = deltaW_HiddenOutput(i,j);
            }
            else{
                deltaHiddenOutput(i,j) = deltaB_HiddenOutput(i,0);
            }
        }
    }
//    cout<<deltaHiddenOutput<<endl;

    NN->wHiddenOutput +=  deltaHiddenOutput;
    NN->wInputHidden += deltaInputHidden;

}

void BatchTrainer::runTrainingEpoch(std::vector<DataEntry *> trainingSet) {
    //incorrect patterns
    double incorrectPatterns = 0;

    //for every training pattern
    for ( int tp = 0; tp < (int) trainingSet.size(); tp++) {

        //feed inputs through network and backpropagate errors
        NN->feedForwardPattern( trainingSet[tp]->pattern );
        backpropagate( trainingSet[tp]->target, tp );

        //pattern correct flag
        bool patternCorrect = true;

        //check all outputs from neural network against desired values
        //pattern incorrect if desired and output differ
        patternCorrect = Trainer::checkOutput(NN->clampOutput(), trainingSet[tp]->target);

        //if pattern is incorrect add to incorrect count
        if ( !patternCorrect ) incorrectPatterns++;

    }//end for

    //update training accuracy and MSE
    trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
}

void BatchTrainer::trainNetwork( trainingDataSet* tSet ){
    cout	<< endl << " Neural Network Training Starting: " << endl
            << "==========================================================================" << endl
            << " LR: " << learningRate << ", Max Epochs: " << maxEpochs << endl
            << " " << NN->nInput << " Input Neurons, " << NN->nHidden << " Hidden Neurons, " << NN->nOutput << " Output Neurons" << endl
            << "==========================================================================" << endl << endl;

    //reset epoch and log counters
    epoch = 0;

    //train network using training dataset for training and generalization dataset for testing
    while (	( trainingSetAccuracy < desiredAccuracy) && epoch < maxEpochs ) {
        //store previous accuracy
        double previousTAccuracy = trainingSetAccuracy;
        double previousGAccuracy = generalizationSetAccuracy;

        //use training set to train network

        runTrainingEpoch( tSet->trainingSet);
        //print out change in training /generalization accuracy (only if a change is greater than a percent)
        if ( ceil(previousTAccuracy) < ceil(trainingSetAccuracy)  || (epoch%10 ==0)) {
            cout << "Epoch : " << epoch <<" trainingSetAccuracy: "<<trainingSetAccuracy<<endl;
        }
        //once training set is complete increment epoch
        epoch++;
    }//end while

    NN->saveWeights("weights.txt");
    validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);

    //log end

    cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
    cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
}