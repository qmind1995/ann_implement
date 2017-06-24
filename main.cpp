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
    NeuralNetwork *nn = new NeuralNetwork(784,10,10);
    string imgFileName = "/home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte";
    string labelFileName = "/home/tri/Desktop/ann_implement/data/t10k-labels.idx1-ubyte";
    DataReader *dR = new DataReader();
    dR->read_Input(imgFileName, labelFileName);
//    arma::mat out = nn->feedForwardPattern(dR->data[0]->pattern);

    //seed random number generator
//    cout<<dR->data[0]->target<<endl;
    srand( (unsigned int) time(0) );

    //create neural network trainer
    Trainer nT( nn );
    trainingDataSet* tSet = new trainingDataSet();
    tSet->trainingSet = dR->data;
    nT.trainNetwork(tSet);
    cout<<"debuger node";
//    nT.setTrainingParameters(0.001, 0.9, false);
//    nT.setStoppingConditions(150, 90);
//    nT.enableLogging("log.csv", 5);
//
//    //train neural network on data sets
//    for (int i=0; i < d.getNumTrainingSets(); i++ )
//    {
//        nT.trainNetwork( d.getTrainingDataSet() );
//    }
//
//    //save the weights
//    nn.saveWeights("weights.csv");
//
//    cout << endl << endl << "-- END OF PROGRAM --" << endl;
//    char c; cin >> c;
    return 0;
}
