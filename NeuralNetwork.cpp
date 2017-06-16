//
// Created by tri on 16/06/2017.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetwork.h"
NeuralNetwork::neuralNetwork(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO){

    inputNeurons = new( double[nInput + 1] );
    for ( int i=0; i < nInput; i++ ){
        inputNeurons[i] = 0;
    }

    inputNeurons[nInput] = -1;

    hiddenNeurons = new( double[nHidden + 1] );
    for ( int i=0; i < nHidden; i++ ){
        hiddenNeurons[i] = 0;
    }

    //create hidden bias neuron
    hiddenNeurons[nHidden] = -1;

    outputNeurons = new( double[nOutput] );
    for ( int i=0; i < nOutput; i++ ){
        outputNeurons[i] = 0;
    }

    //create weight lists (include bias neuron weights)
    wInputHidden = new( double*[nInput + 1] );
    for ( int i=0; i <= nInput; i++ ) {
        wInputHidden[i] = new (double[nHidden]);
        for ( int j=0; j < nHidden; j++ ) wInputHidden[i][j] = 0;
    }

    wHiddenOutput = new( double*[nHidden + 1] );
    for ( int i=0; i <= nHidden; i++ ) {
        wHiddenOutput[i] = new (double[nOutput]);
        for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = 0;
    }

    //initialize weights
    initializeWeights();
}

NeuralNetwork::initializeWeights(){
    //set range
    double rH = 1/sqrt( (double) nInput);
    double rO = 1/sqrt( (double) nHidden);

    //set weights between input and hidden
    for(int i = 0; i <= nInput; i++) {
        for(int j = 0; j < nHidden; j++) {
            //set weights to random values
            wInputHidden[i][j] = ( ( (double)(rand()%100)+1)/100  * 2 * rH ) - rH;
        }
    }

    //set weights between input and hidden
    for(int i = 0; i <= nHidden; i++) {
        for(int j = 0; j < nOutput; j++) {
            //set weights to random values
            wHiddenOutput[i][j] = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;
        }
    }
}