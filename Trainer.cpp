//
// Created by tri on 23/06/2017.
//

#include "Trainer.h"
Trainer::Trainer( NeuralNetwork *nn )	:	NN(nn),
                                            epoch(0),
                                            learningRate(LEARNING_RATE),
                                            desiredAccuracy(DESIRED_ACCURACY),
                                            maxEpochs(MAX_EPOCHS),
                                            trainingSetAccuracy(0),
                                            validationSetAccuracy(0),
                                            generalizationSetAccuracy(0),
                                            trainingSetMSE(0),
                                            validationSetMSE(0),
                                            generalizationSetMSE(0) {
    //create delta lists
    deltaInputHidden = mat(nn->nHidden, nn->nInput + 1);
    deltaInputHidden.zeros();

    deltaHiddenOutput = mat(nn->nOutput, nn->nHidden +1);
    deltaHiddenOutput.zeros();

    //create error gradient storage
    //--------------------------------------------------------------------------------------------------------
    hiddenErrorGradients = mat(nn->nHidden +1,1);
    hiddenErrorGradients.zeros();

    outputErrorGradients = mat(nn->nOutput +1,1);
    outputErrorGradients.zeros();

}

mat Trainer::dotProduct(mat A, mat B) {
    auto sizeA = arma::size(A);
    auto sizeB = arma::size(B);
    if(sizeA.n_rows != sizeB.n_rows || sizeA.n_cols != sizeB.n_cols){
        cout<<"dont match size A and B ! check again idot!"<<endl;
    }
    mat output = mat(sizeA.n_rows, sizeA.n_cols);
    for(int i =0; i< sizeA.n_rows; i++){
        for(int j=0; j< sizeA.n_cols; j++){
            output(i,j) = A(i,j) * B(i,j);
        }
    }

    return output;
}

inline mat Trainer::getOutputErrorGradient( mat desiredValue, mat outputValue) {
    //return error gradient
    mat tmp, first;
    if(NN->activationFuncName == "SIGMOID"){
        tmp =  1 - outputValue ;
        first = dotProduct(outputValue, tmp);
    }
    else if(NN->activationFuncName == "TANH"){
        first = 1 - dotProduct(outputValue, outputValue);
    }

    mat err = desiredValue - outputValue;

    return dotProduct(first, err);
}

mat Trainer::getHiddenErrorGradient(mat outErGradients) {
    mat hiddenNoBias(NN->nHidden,1);
    for(int i=0; i<NN->nHidden; i++){
        hiddenNoBias(i,0) = NN->hiddenNeurons(i,0);
    }

    mat first;

    if(NN->activationFuncName == "SIGMOID"){
        mat tmp = 1 - hiddenNoBias;
        first = dotProduct(hiddenNoBias, tmp);
    }
    else if(NN->activationFuncName == "TANH"){
        first = 1 - dotProduct(hiddenNoBias, hiddenNoBias);
    }

    mat wHO_NoBias(NN->nOutput, NN->nHidden);

    for(int i=0; i<NN->nOutput; i++){
        for(int j=0; j< NN->nHidden; j++){
            wHO_NoBias(i,j) = NN->wHiddenOutput(i,j);
        }
    }

    mat err = wHO_NoBias.t() * outErGradients;
    mat output = dotProduct(first,err);
    return output;
}

void Trainer::backpropagate( mat desiredOutputs ){
    //modify deltas between hidden and output layers

    outputErrorGradients = getOutputErrorGradient(desiredOutputs, NN->outputNeurons);
    deltaHiddenOutput = learningRate * outputErrorGradients * NN->hiddenNeurons.t();

    hiddenErrorGradients = getHiddenErrorGradient(outputErrorGradients);
    deltaInputHidden = learningRate * hiddenErrorGradients * NN->inputNeurons.t();

    updateWeights();
}

void Trainer::updateWeights() {
    NN->wHiddenOutput += deltaHiddenOutput;
    NN->wInputHidden += deltaInputHidden;
}

bool Trainer::checkOutput(mat output, mat target){
    int size = NN->nOutput;
    for(int i =0; i<size; i++){
        if(abs(output(i,0) - target(i,0) ) > 0.1){
            return false;
        }
    }
    return true;
}

void Trainer::runTrainingEpoch( vector<DataEntry*> trainingSet ) {
    //incorrect patterns
    double incorrectPatterns = 0;
    double mse = 0;

    //for every training pattern
    for ( int tp = 0; tp < (int) trainingSet.size(); tp++) {

        //feed inputs through network and backpropagate errors
        NN->feedForwardPattern( trainingSet[tp]->pattern );
        backpropagate( trainingSet[tp]->target );

        //pattern correct flag
        bool patternCorrect = true;

        //check all outputs from neural network against desired values
        //pattern incorrect if desired and output differ
        patternCorrect = checkOutput(NN->clampOutput(), trainingSet[tp]->target);

        //if pattern is incorrect add to incorrect count
        if ( !patternCorrect ) incorrectPatterns++;

    }//end for

    //update training accuracy and MSE
    trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
}

void Trainer::trainNetwork( trainingDataSet* tSet ) {
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

        runTrainingEpoch( tSet->trainingSet );
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