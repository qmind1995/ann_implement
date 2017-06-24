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
    mat tmp =  1 - outputValue ;
    mat first = dotProduct(outputValue, tmp);
    mat err = desiredValue - outputValue;

    return dotProduct(first, err);
}

mat Trainer::getHiddenErrorGradient() {
    mat tmp = 1 - NN->hiddenNeurons;
    mat first = dotProduct(NN->hiddenNeurons, tmp);
    mat err = NN->wHiddenOutput.t() * outputErrorGradients;
    mat preOutput = dotProduct(first,err);
    mat output = mat(NN->nHidden,1);
    output.zeros();

    for(int i=0; i<NN->nHidden; i++){
        output(i,0) = preOutput(i,0);

    }
    return output;
}

void Trainer::backpropagate( mat desiredOutputs ){
    //modify deltas between hidden and output layers

    outputErrorGradients = getOutputErrorGradient(desiredOutputs, NN->outputNeurons);
    deltaHiddenOutput = learningRate * outputErrorGradients * NN->hiddenNeurons.t();

    hiddenErrorGradients = getHiddenErrorGradient();
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
//    cout<<trainingSetAccuracy<<endl;
}

void Trainer::trainNetwork( trainingDataSet* tSet ) {
    cout	<< endl << " Neural Network Training Starting: " << endl
            << "==========================================================================" << endl
            << " LR: " << learningRate << ", Max Epochs: " << maxEpochs << endl
            << " " << NN->nInput << " Input Neurons, " << NN->nHidden << " Hidden Neurons, " << NN->nOutput << " Output Neurons" << endl
            << "==========================================================================" << endl << endl;

    //reset epoch and log counters
    epoch = 0;
    lastEpochLogged = -logResolution;
    //runTrainingEpoch( tSet->trainingSet );

    //train network using training dataset for training and generalization dataset for testing

    while (	( trainingSetAccuracy < desiredAccuracy) && epoch < maxEpochs ) {
        //store previous accuracy
        double previousTAccuracy = trainingSetAccuracy;
        double previousGAccuracy = generalizationSetAccuracy;

        //use training set to train network
        runTrainingEpoch( tSet->trainingSet );

        //get generalization set accuracy and MSE
//        generalizationSetAccuracy = NN->getSetAccuracy( tSet->generalizationSet );
        //generalizationSetMSE = NN->getSetMSE( tSet->generalizationSet );

        //Log Training results
//        if ( loggingEnabled && logFile.is_open() && ( epoch - lastEpochLogged == logResolution ) ) {
//            logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl;
//            lastEpochLogged = epoch;
//        }

        //print out change in training /generalization accuracy (only if a change is greater than a percent)
        if ( ceil(previousTAccuracy) < ceil(trainingSetAccuracy)  || (epoch%10 ==0)) {
            cout << "Epoch : " << epoch <<" trainingSetAccuracy: "<<trainingSetAccuracy<<endl;
            //cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
            //cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;
        }

        //once training set is complete increment epoch
        epoch++;

    }//end while

    //get validation set accuracy and MSE
    validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);
    //validationSetMSE = NN->getSetMSE(tSet->validationSet);

    //log end
    logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl << endl;
    logFile << "Training Complete!!! - > Elapsed Epochs: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << " Validation Set MSE: " << validationSetMSE << endl;

    //out validation accuracy and MSE
    cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
    cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
//    cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
}