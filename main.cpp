//
// Created by tri on 16/06/2017.
//
#include<iostream>
#include <ctime>
#include "NeuralNetwork.h"
#include "Trainer.h"

using namespace std;
using namespace arma;
int main() {
//    NeuralNetwork *nn = new NeuralNetwork(784,10,10);
    NeuralNetwork *nn = new NeuralNetwork("weights.txt");
    string imgFileName = "/home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte";
    string labelFileName = "/home/tri/Desktop/ann_implement/data/t10k-labels.idx1-ubyte";
    DataReader *dR = new DataReader();
    dR->read_Input(imgFileName, labelFileName);
    srand( (unsigned int) time(0) );

    //create neural network trainer
    Trainer nT( nn );
    trainingDataSet* tSet = new trainingDataSet();
    tSet->trainingSet = dR->data;
    nT.trainNetwork(tSet);

    return 0;
}
