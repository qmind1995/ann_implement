//
// Created by tri on 16/06/2017.
//
#include<iostream>
#include "NeuralNetwork.h"
#include "Trainer.h"
#include "BatchTrainer.h"
#include "DataGenerator.h"

using namespace std;
using namespace arma;
//
//int main() {
//    NeuralNetwork *nn = new NeuralNetwork(784,100,10, "SIGMOID");
////    NeuralNetwork *nn = new NeuralNetwork("weights.txt","TANH");
//    string imgFileName_test = "/home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte";
//    string labelFileName_test = "/home/tri/Desktop/ann_implement/data/t10k-labels.idx1-ubyte";
//    string imgFileName = "/home/tri/Desktop/ann_implement/data/train-images.idx3-ubyte";
//    string labelFileName = "/home/tri/Desktop/ann_implement/data/train-labels.idx1-ubyte";
//    DataReader *dR = new DataReader();
//    dR->read_Input(imgFileName, labelFileName, 60000);
//    srand( (unsigned int) time(0) );
//
//    //create neural network trainer
//    Trainer nT( nn );
//    trainingDataSet* trSet = new trainingDataSet();
////    BatchTrainer nT(nn, 100);
////    trainingDataSet* trSet = new trainingDataSet();
//    trSet->trainingSet = dR->data;
//
//
//    DataReader *dR_test = new DataReader();
//    dR_test->read_Input(imgFileName_test, labelFileName_test, 10000);
//    trSet->validationSet = dR_test->data;
//
//    nT.trainNetwork(trSet);
//    return 0;
//}

int main(){
    int shouldGenFile = false;
    string outputDataFileName = "/home/tri/Desktop/ann_implement/data/sinData.txt";
    string outputTestFileName = "/home/tri/Desktop/ann_implement/data/sinData_test.txt";
    string inputDataFileName = "/home/tri/Desktop/ann_implement/data/sinInput.txt";
    string inputtestFileName = "/home/tri/Desktop/ann_implement/data/sinInput_test.txt";
    cout<<"If you want to gen data files for sin method, press 1! else press 0."<<endl;
    cin>>shouldGenFile;
    if(shouldGenFile == 1){
        DataGenerator *dataGen = new DataGenerator();
        dataGen->genDataForSinFunction(inputDataFileName, outputDataFileName, 60000);
        dataGen->genDataForSinFunction(inputtestFileName, outputTestFileName, 10000);
        cout<<"gen data done!"<<endl;
    }

    DataReader *dR = new DataReader();
    dR->read_RegressionData(inputDataFileName, outputDataFileName, 10000);

    NeuralNetwork *nn = new NeuralNetwork(1,20,1, "TANH");

    Trainer nT( nn );
//    BatchTrainer nT(nn, 100);

    trainingDataSet* trSet = new trainingDataSet();
    trSet->trainingSet = dR->data;

    DataReader *dR_test = new DataReader();
    dR_test->read_RegressionData(inputtestFileName, outputTestFileName, 3000);
    trSet->validationSet = dR_test->data;

    cout<<"============================= START TRAINING ============================="<<endl;

    nT.trainNetwork(trSet);

    return 0;
}