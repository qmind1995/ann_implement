//
// Created by tri on 16/06/2017.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "NeuralNetwork.h"
using namespace arma;

NeuralNetwork::NeuralNetwork(vector<Layer *> nlayers):layers(nlayers) {

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

void NeuralNetwork::updateWeights(vector<mat> deltaWeights, vector<mat> deltaBiass) {

    for(int i= 0; i < nLayer -1; i++){
        weights[i]  = weights[i] + deltaWeights[i];

        if(deltaBiass[i].n_rows != 0){
            biass[i] = biass[i] + deltaBiass[i];
        }
    }
}

/*

bool NeuralNetwork::saveWeights(string filename) {

    fstream outputFile;
    outputFile.open(filename, ios::out);

    if ( outputFile.is_open() ) {

        outputFile<<nInput<<"\n";
        outputFile<<nHidden<<"\n";
        outputFile<<nOutput<<"\n";

        // save Input-> Hidden weights:
        for(int i=0; i< nHidden; i++){
            for(int j=0; j< nInput+1; j++){
                outputFile<<wInputHidden(i,j) <<"\t";
            }
            outputFile<<"\n";
        }

        // save Hidden -> Output weights:
        for(int i=0; i< nOutput; i++){
            for(int j=0; j< nHidden+1; j++){
                outputFile<<wHiddenOutput(i,j)<<"\t";
            }
            outputFile<<"\n";
        }

        //print success
        cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

        //close file
        outputFile.close();

        return true;
    }
    else {
        cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
        return false;
    }
}*/

//implement new version:
