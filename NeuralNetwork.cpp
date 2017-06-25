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
NeuralNetwork::NeuralNetwork(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO){

    inputNeurons = mat(nInput + 1, 1);
    for ( int i=0; i < nInput; i++ ){
        inputNeurons(i,0) =0;
    }

    //bias
    inputNeurons(nInput,0) = -1;

    hiddenNeurons = mat(nHidden + 1, 1);
    for ( int i=0; i < nHidden; i++ ){
        hiddenNeurons(i,0) =0;
    }

    //create hidden bias neuron
    hiddenNeurons(nHidden,0) = -1;

    outputNeurons = mat(nOutput, 1);
    for ( int i=0; i < nOutput; i++ ){
        outputNeurons(i,0) =0;
    }

    //create weight lists (include bias neuron weights)
//    wInputHidden = mat(nInput + 1, nHidden);
    wInputHidden = mat(nHidden, nInput + 1);
    for ( int i=0; i < nHidden; i++ ) {
        for ( int j=0; j <= nInput; j++ ){
            wInputHidden(i,j) = 0;
        }
    }

//    wHiddenOutput = mat(nHidden + 1, nOutput);
    wHiddenOutput = mat(nOutput, nHidden + 1);
    for ( int i=0; i < nOutput; i++ ) {
        for ( int j=0; j <= nHidden; j++ ){
            wHiddenOutput(i,j) = 0;
        }
    }

    //initialize weights
    initializeWeights();
}

void NeuralNetwork::initializeWeights(){
    //set range
    double rH = 1/sqrt( (double) nInput);
    double rO = 1/sqrt( (double) nHidden);

    //set weights between input and hidden
    for ( int i=0; i < nHidden; i++ ) {
        for ( int j=0; j <= nInput; j++ ){
            //set weights to random values
            wInputHidden(i,j) = ( ( (double)(rand()%100)+1)/100  * 2 * rH ) - rH;
        }
    }

    //set weights between input and hidden
    for ( int i=0; i < nOutput; i++ ) {
        for ( int j=0; j <= nHidden; j++ ){
            //set weights to random values
            wHiddenOutput(i,j) = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;
        }
    }
}

inline double NeuralNetwork::activationFunction( double x ) {
    //sigmoid function
    return 1/(1+exp(-x));
}

void NeuralNetwork::feedForward(mat input) {

    //clone input to input neurals.
    for(int i=0; i< nInput ; i++){
        inputNeurons(i,0) = input(i,0);
    }

    //Calculate Hidden Layer values - include bias neuron

    mat preHidden = wInputHidden * inputNeurons; //not bias yet.!

    // use activation function:

    for(int i =0; i< nHidden; i++){
        hiddenNeurons(i,0) = activationFunction(preHidden(i,0));
    }

    //Calculate output

    mat preOutput = wHiddenOutput * hiddenNeurons; // not bias yet!

    //use activation function:

    for(int i=0; i < nOutput; i++){
        outputNeurons(i,0) = activationFunction(preOutput(i,0));
    }

}

mat NeuralNetwork::clampOutput(){
    mat res =mat(nOutput,1);
    for(int i=0 ;i< nOutput; i++){
        if(outputNeurons(i,0) < 0.4){
            res(i,0) =0;
        }
        else if(outputNeurons(i,0) >=0.6){
            res(i,0) =1;
        }
        else{
            res(i,0) = outputNeurons(i,0);
        }
    }
    return res;
}

mat NeuralNetwork::feedForwardPattern(mat input){
    feedForward(input);
    return outputNeurons;
}
bool NeuralNetwork::checkOutput(mat output, mat target){
    int size = nOutput;
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
    for ( int tp = 0; tp < (int) set.size(); tp++)
    {
        //feed inputs through network and backpropagate errors
        feedForward( set[tp]->pattern );

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

bool NeuralNetwork::saveWeights(char* filename) {

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
}
//
//int main(){
//    NeuralNetwork *nn = new NeuralNetwork(784,10,10);
//    string imgFileName = "/home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte";
//    string labelFileName = "/home/tri/Desktop/ann_implement/data/t10k-labels.idx1-ubyte";
//    DataReader *dR = new DataReader();
//    dR->read_Input(imgFileName, labelFileName);
//    mat out = nn->feedForwardPattern(dR->data[0]->pattern);
//    cout<<out;
//
//    return 0;
//}