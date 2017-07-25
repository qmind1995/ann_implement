//
// Created by tri on 16/06/2017.
//
#include<iostream>
#include "ann/NeuralNetwork.h"
#include "ann/Trainer.h"
#include "ann/BatchTrainer.h"
#include "dataReader/DataGenerator.h"
#include "ann/Layer.h"
#include "visualize/Visualize.cpp"

#define N_THREAD   2

using namespace std;
using namespace arma;
using namespace parameters;

void * runtrainingThread(void * nNet){
    string outputDataFileName = "/home/tri/Desktop/ann_implement/data/sinData.txt";
    string outputTestFileName = "/home/tri/Desktop/ann_implement/data/sinData_test.txt";
    string inputDataFileName = "/home/tri/Desktop/ann_implement/data/sinInput.txt";
    string inputtestFileName = "/home/tri/Desktop/ann_implement/data/sinInput_test.txt";

    DataReader *dR = new DataReader();
    dR->read_RegressionData(inputDataFileName, outputDataFileName, 60000);
    NeuralNetwork* net = reinterpret_cast<NeuralNetwork*>(nNet);
    Trainer * nT = new BatchTrainer( net,100);
//    Trainer * nT = new Trainer( net);

    trainingDataSet* trSet = new trainingDataSet();
    trSet->trainingSet = dR->data;

    DataReader *dR_test = new DataReader();
    dR_test->read_RegressionData(inputtestFileName, outputTestFileName, 3000);
    trSet->validationSet = dR_test->data;

    nT->trainNetwork(trSet);
    pthread_exit(NULL);
}

void* runVisualizeThread(void * params){
    struct paramHolder * ps = reinterpret_cast<struct paramHolder *>(params);
    visualize(ps->net, ps->argc, ps->argv);

    pthread_exit(NULL);
}

int main(int argc, char** argv){
    Layer* inputLayer = new Layer(1, true, parameters::NONE);
    Layer* hiddenLayer = new Layer(20, true, parameters::TANH);
    Layer* outputLayer = new Layer(1, false, parameters::TANH);
    vector<Layer*> layers;
    layers.push_back(inputLayer);
    layers.push_back(hiddenLayer);
    layers.push_back(outputLayer);

    NeuralNetwork * nNet = new NeuralNetwork(layers, REGRESSTION);

    pthread_t threads[N_THREAD];

    int rc1 = pthread_create(&threads[0], NULL,runtrainingThread,(void*) nNet);
    if (rc1){
        cout << "\nError: Khong the tao training thread! " << rc1 << endl;
        exit(-1);
    }

    struct paramHolder * params = new paramHolder();
    params->argc = argc;
    params->argv = argv;
    params->net = nNet;

    int rc2 = pthread_create(&threads[1], NULL,runVisualizeThread,(void*) params);
    if (rc2){
        cout << "\nError: Khong the tao drawing thread! " << rc2 << endl;
        exit(-1);
    }

    pthread_exit(NULL);
    return 0;
}
