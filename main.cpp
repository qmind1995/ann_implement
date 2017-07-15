//
// Created by tri on 16/06/2017.
//
#include<iostream>
#include <ctime>
#include "NeuralNetwork.h"
#include "Trainer.h"
#include "BatchTrainer.h"

using namespace std;
using namespace arma;
int main() {
    NeuralNetwork *nn = new NeuralNetwork(784,100,10, "SIGMOID");
//    NeuralNetwork *nn = new NeuralNetwork("weights.txt","TANH");
    string imgFileName_test = "/home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte";
    string labelFileName_test = "/home/tri/Desktop/ann_implement/data/t10k-labels.idx1-ubyte";
    string imgFileName = "/home/tri/Desktop/ann_implement/data/train-images.idx3-ubyte";
    string labelFileName = "/home/tri/Desktop/ann_implement/data/train-labels.idx1-ubyte";
    DataReader *dR = new DataReader();
    dR->read_Input(imgFileName, labelFileName, 60000);
    srand( (unsigned int) time(0) );

    //create neural network trainer
    Trainer nT( nn );
    trainingDataSet* trSet = new trainingDataSet();
//    BatchTrainer nT(nn, 100);
//    trainingDataSet* trSet = new trainingDataSet();
    trSet->trainingSet = dR->data;


    DataReader *dR_test = new DataReader();
    dR_test->read_Input(imgFileName_test, labelFileName_test, 10000);
    trSet->validationSet = dR_test->data;

    nT.trainNetwork(trSet);
    return 0;
}
