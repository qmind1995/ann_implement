//
// Created by tri on 16/06/2017.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO){

    inputNeurons = Mat(nInput + 1, 1, CV_64F);
    for ( int i=0; i < nInput; i++ ){
        inputNeurons.at<double>(i,0) =0;
    }

    //bias
    inputNeurons.at<double>(nInput,0) = -1;

    hiddenNeurons = Mat(nHidden + 1, 1, CV_64F);
    for ( int i=0; i < nHidden; i++ ){
        hiddenNeurons.at<double>(i,0) =0;
    }

    //create hidden bias neuron
    hiddenNeurons.at<double>(nHidden,0) = -1;

    outputNeurons = Mat(nOutput, 1, CV_64F);
    for ( int i=0; i < nOutput; i++ ){
        outputNeurons.at<double>(i,0) =0;
    }

    //create weight lists (include bias neuron weights)
    wInputHidden = Mat(nInput + 1, nHidden, CV_64F);
    for ( int i=0; i <= nInput; i++ ) {
        for ( int j=0; j < nHidden; j++ ){
            wInputHidden.at<double>(i,j) = 0;
        }
    }

    wHiddenOutput = Mat(nHidden + 1, nOutput, CV_64F);
    for ( int i=0; i <= nHidden; i++ ) {
        for ( int j=0; j < nHidden; j++ ){
            wHiddenOutput.at<double>(i,j) = 0;
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
    for(int i = 0; i <= nInput; i++) {
        for(int j = 0; j < nHidden; j++) {
            //set weights to random values
            wInputHidden.at<double>(i,j) = ( ( (double)(rand()%100)+1)/100  * 2 * rH ) - rH;
        }
    }

    //set weights between input and hidden
    for(int i = 0; i <= nHidden; i++) {
        for(int j = 0; j < nOutput; j++) {
            //set weights to random values
            wHiddenOutput.at<double>(i,j) = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;
        }
    }
}

inline double NeuralNetwork::activationFunction( double x ) {
    //sigmoid function
    return 1/(1+exp(-x));
}

void NeuralNetwork::feedForward(Mat input) {

    //clone input to input neurals.
    for(int i=0; i< nInput ; i++){
        inputNeurons.at<double>(i,0) = input.at<double>(i,0);
    }

    //Calculate Hidden Layer values - include bias neuron

    Mat preHidden = wInputHidden * inputNeurons; //not bias yet.!

    // use activation function:

    for(int i =0; i< nHidden; i++){
        hiddenNeurons.at<double>(i,0) = activationFunction(preHidden.at<double>(i,0));
    }

    //Calculate output

    Mat preOutput = wHiddenOutput * hiddenNeurons; // not bias yet!

    //use activation function:

    for(int i=0; i < nOutput; i++){
        outputNeurons.at<double>(i,0) = activationFunction(preOutput.at<double>(i,0));
    }

}

inline Mat NeuralNetwork::clampOutput(){
    Mat output(nOutput, 1, CV_64F);
//    for(int i =0 ; i<nOutput; i++){
//        outputNeurons.at<double>(i, 0);
//    }

    //no need this function now.
}

Mat NeuralNetwork::feedForwardPattern(Mat input){
    feedForward(input);
    return outputNeurons;
}

int main(){
    NeuralNetwork *nn = new NeuralNetwork(784,10,10);
    string imgFileName = "/home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte";
    String labelFileName = "/home/tri/Desktop/ann_implement/data/t10k-labels.idx1-ubyte";
//    int number_of_images = 10000;
//    int image_size = 28 * 28;
    DataReader *dR = new DataReader();
    dR->read_Input(imgFileName, labelFileName);
    cout<<dR->data[0]->pattern;
    Mat out = nn->feedForwardPattern(dR->data[0]->pattern);


    return 0;
}