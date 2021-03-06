//
// Created by tri on 16/06/2017.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(vector<Layer *> nlayers, int type):layers(nlayers), netType(type) {

    nLayer = (int)layers.size();
    for(int i =0; i < nLayer -1 ; i++){ // there are n-1 W between n layers
        int layerSize = layers[i]->nNeurals;
        int nextLayerSize = layers[i+1]->nNeurals;

        mat weight = initializeWeights(nextLayerSize, layerSize);
        weights.push_back(weight);

        mat bias;
        if(layers[i]->isBias){
            bias = initializeWeights(nextLayerSize, 1);
        }

        // if this layer has no bias => push empty(size = [0x0]); and Output layer has no bias.
        biass.push_back(bias);
    }

}

NeuralNetwork::NeuralNetwork(string weightFileName){
    ifstream weightFileStream(weightFileName);
    if(!weightFileStream.is_open()){
        cout<<"cannot open this file ! idiot !.\n";
    }

}

mat NeuralNetwork::initializeWeights(int nRows, int nCols){
    mat weight;
    weight = mat(nRows, nCols);
    weight.zeros();

    for(int i=0; i< nRows; i++){
        for(int j=0; j< nCols; j++){
            weight(i, j) = gaussianRamdom(0, 0.5);
        }
    }

    return weight;
}

mat NeuralNetwork::feedForwardPattern(mat input){

    layers[0]->setNeuralsValue(input);
    layers[0]->activation();

    for(int i=1; i < nLayer; i++){

        mat tmp = weights[i-1] * layers[i-1]->neurals;
        if(biass[i - 1].n_cols != 0){
            tmp = tmp+ biass[i -1];
        }

        layers[i]->setNeuralsValue(tmp);
        layers[i]->activation();
    }
    return layers[nLayer - 1]->neurals;
}


mat NeuralNetwork::clampOutput(){ // this function is applied to classification

    mat res = layers[nLayer - 1]->neurals; // output layer

    for(int i=0 ;i< res.n_rows; i++){
        if(res(i,0) < 0.5){
            res(i,0) =0;
        }
        else if(res(i,0) >=0.5){
            res(i,0) =1;
        }
    }
    return res;
}

bool NeuralNetwork::checkOutput(mat output , mat target){
    int size = output.n_rows;
    for(int i =0; i<size; i++){
        if(abs(output(i,0) - target(i,0) ) > 0.001){
            return false;
        }
    }
    return true;
}

double NeuralNetwork::getSetAccuracy( std::vector<DataEntry*>& set ) {
    double incorrectResults = 0;

    //for every training input array
    int size = (int)set.size();
    for ( int tp = 0; tp < size; tp++) {
        //feed inputs through network and backpropagate errors
        feedForwardPattern( set[tp]->pattern );

        //correct pattern flag
        bool correctResult = true;

        //check all outputs against desired output values
        correctResult = checkOutput(clampOutput(), set[tp]->target);

        //inc training error for a incorrect result
        if ( !correctResult ) incorrectResults++;

    }//end for

    //calculate error and return as percentage
    return 100 - (incorrectResults/set.size() * 100);
}

void NeuralNetwork::updateWeights(vector<mat> deltaWeights, vector<mat> deltaBiass, double learningRate) {

    for(int i= 0; i < nLayer -1; i++){
        weights[i]  += learningRate * deltaWeights[i];

        if(deltaBiass[i].n_rows != 0){
            biass[i] += learningRate * deltaBiass[i];
        }
    }
}

mat NeuralNetwork::getOutput() {
    return layers[nLayer -1]->neurals;
}

mat NeuralNetwork::getVisualizeOutput(mat input) {

    mat tmpNeurals = input;

    for(int i=1; i < nLayer; i++){
        mat tmp = weights[i-1] * tmpNeurals;
        if(biass[i - 1].n_cols != 0){
            tmp = tmp+ biass[i -1];
        }

        tmpNeurals = activationFunction(tmp, layers[i]->activeFunc);
    }

    return tmpNeurals;

}

vector<string> NeuralNetwork::getNeuralInfoForVisualize() {
    vector<string> output;
    output.push_back("Network infomation: ");
    if(netType == CLASSIFICATION){
        output.push_back("Network type: classification");
    }
    else{
        output.push_back("Network type: regression");
    }
    output.push_back("nlayer: " +to_string(nLayer) );
    for(int i =0; i < nLayer; i++){
        string layerInfo = "Layer " + to_string(i+1) + " has " + to_string(layers[i]->nNeurals) + " neurals; ";
        layerInfo += "activation function: " + activeFuncNameToString(layers[i]->activeFunc);
        output.push_back(layerInfo);
    }
    return output;
}

void NeuralNetwork::printNetwokInfo() {
    cout <<"Network infomation: "<<endl;
    for(int i =0; i < nLayer; i++){
        cout<<"Layer "<<i +1 << " has "<<layers[i]->nNeurals <<" neurals;  "<<"activation function: ";
        switch(layers[i]->activeFunc){
            case TANH: {
                cout << "Tanh" << endl;
                break;
            }
            case SIGMOID: {
                cout << "Sigmoid" << endl;
                break;
            }
            case RELU: {
                cout << "Relu" << endl;
                break;
            }
            case parameters::NONE: {
                cout << "None" << endl;
                break;
            }
        }
    }
    cout<<"=========================================================================="<<endl<<endl;
}
