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
    int nLayers = nn->nLayer;
    for(int i=0; i< nLayers - 1; i++){
        int layerSize = nn->layers[i]->nNeurals;
        int nextLayerSize = nn->layers[i+1]->nNeurals;

        mat deltaW = mat(nextLayerSize, layerSize);
        deltaW.zeros();
        deltaWeights.push_back(deltaW);

        mat deltaB;
        if(nn->layers[i]->isBias){
            deltaB = mat(nextLayerSize, 1);
            deltaB.zeros();
        }

        // if this layer has no bias => push empty(size = [0x0]); and Output layer has no bias.
        deltaBiass.push_back(deltaB);
    }

    //create error gradient storage
    //------------------------------------------
    for(int i=1; i< nLayers; i++){
        mat eG = mat(nn->layers[i]->nNeurals, 1);
        eG.zeros();
        errorGradients.push_back(eG);
    }
}

void Trainer::backpropagate( mat desiredOutputs ){

    //modify deltas between layers
    int nLayers = NN->nLayer;

    mat err = desiredOutputs - NN->layers[nLayers-1]->neurals;

    for(int i=nLayers-1; i >=0; i--){

        mat errorGradient = NN->layers[nLayers-1]->getGradient(err);
        mat deltaW = errorGradient * NN->layers[nLayers-1]->neurals;
        deltaWeights.push_back(deltaW);

        mat deltaB;

        if( NN->biass[i-1].n_rows != 0 ){
            deltaB = errorGradient;
        }
        deltaBiass.push_back(deltaB);
        err = NN->weights[i-1] .t() * errorGradient;
    }
    reverse(deltaWeights.begin(), deltaWeights.end());
    reverse(deltaBiass.begin(), deltaBiass.end());

    updateWeights();
}

void Trainer::updateWeights() {
    NN->updateWeights(deltaWeights, deltaBiass);
}

//void Trainer::runTrainingEpoch( vector<DataEntry*> trainingSet ) {
//    //incorrect patterns
//    double incorrectPatterns = 0;
//    double mse = 0;
//
//    //for every training pattern
//    for ( int tp = 0; tp < (int) trainingSet.size(); tp++) {
//
//        //feed inputs through network and backpropagate errors
//        NN->feedForwardPattern( trainingSet[tp]->pattern );
//        backpropagate( trainingSet[tp]->target );
//
//        //pattern correct flag
//        bool patternCorrect = true;
//
//        //check all outputs from neural network against desired values
//        //pattern incorrect if desired and output differ
//        patternCorrect = checkOutput(NN->clampOutput(), trainingSet[tp]->target);
//
//        //if pattern is incorrect add to incorrect count
//        if ( !patternCorrect ) incorrectPatterns++;
//
//        mse += (NN->outputNeurons(0,0) - trainingSet[tp]->target(0,0)) * (NN->outputNeurons(0,0) - trainingSet[tp]->target(0,0));
//
//    }//end for
//
//    //update training accuracy and MSE
//    trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
//    trainingSetMSE = mse/trainingSet.size();
//}

//bool Trainer::checkOutput(mat output, mat target){
//    int size = NN->nOutput;
//    for(int i =0; i<size; i++){
//        if(abs(output(i,0) - target(i,0) ) > 0.1){
//            return false;
//        }
//    }
//    return true;
//}
//
//void Trainer::runTrainingEpoch( vector<DataEntry*> trainingSet ) {
//    //incorrect patterns
//    double incorrectPatterns = 0;
//    double mse = 0;
//
//    //for every training pattern
//    for ( int tp = 0; tp < (int) trainingSet.size(); tp++) {
//
//        //feed inputs through network and backpropagate errors
//        NN->feedForwardPattern( trainingSet[tp]->pattern );
//        backpropagate( trainingSet[tp]->target );
//
//        //pattern correct flag
//        bool patternCorrect = true;
//
//        //check all outputs from neural network against desired values
//        //pattern incorrect if desired and output differ
//        patternCorrect = checkOutput(NN->clampOutput(), trainingSet[tp]->target);
//
//        //if pattern is incorrect add to incorrect count
//        if ( !patternCorrect ) incorrectPatterns++;
//
//        mse += (NN->outputNeurons(0,0) - trainingSet[tp]->target(0,0)) * (NN->outputNeurons(0,0) - trainingSet[tp]->target(0,0));
//
//    }//end for
//
//    //update training accuracy and MSE
//    trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
//    trainingSetMSE = mse/trainingSet.size();
//}
//
//void Trainer::trainNetwork( trainingDataSet* tSet ) {
//    cout	<< endl << " Neural Network Training Starting: " << endl
//            << "==========================================================================" << endl
//            << " LR: " << learningRate << ", Max Epochs: " << maxEpochs << endl
//            << " " << NN->nInput << " Input Neurons, " << NN->nHidden << " Hidden Neurons, " << NN->nOutput << " Output Neurons" << endl
//            << "==========================================================================" << endl << endl;
//
//    //reset epoch and log counters
//    epoch = 0;
//
//    //train network using training dataset for training and generalization dataset for testing
//
//    while (	( trainingSetAccuracy < desiredAccuracy) && epoch < maxEpochs ) {
//        //store previous accuracy
//        double previousTAccuracy = trainingSetAccuracy;
//        double previousGAccuracy = generalizationSetAccuracy;
//
//        //use training set to train network
//
//        runTrainingEpoch( tSet->trainingSet );
//        //print out change in training /generalization accuracy (only if a change is greater than a percent)
//        if ( ceil(previousTAccuracy) < ceil(trainingSetAccuracy)  || (epoch%10 ==0)) {
//            cout << "Epoch : " << epoch <<" trainingSetAccuracy: "<<trainingSetAccuracy<<endl;
//        }
//        cout << "Epoch : " << epoch <<" trainingSetMSE: "<<trainingSetMSE<<endl;
//        //once training set is complete increment epoch
//        epoch++;
//
//    }//end while
//
//    NN->saveWeights("weights.txt");
//    validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);
//
//    //log end
//
//    cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
//    cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
//}

int main(){
    Layer* inputLayer = new Layer(1, true, constant::NONE);
    Layer* hiddenLayer = new Layer(10, true, constant::TANH);
    Layer* outputLayer = new Layer(1, false, constant::TANH);
    vector<Layer*> layers;
    layers.push_back(inputLayer);
    layers.push_back(hiddenLayer);
    layers.push_back(outputLayer);

    NeuralNetwork * nNet = new NeuralNetwork(layers);
    Trainer *tn = new Trainer(nNet);
    mat data(1,1);
    data(0,0) =1;

    nNet->feedForwardPattern(data);
    nNet->clampOutput();
    return 0;
}