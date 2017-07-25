//
// Created by tri on 15/07/2017.
//

#include "DataGenerator.h"


void DataGenerator::genDataForSinFunction(string inputDataFileName, string outputDataFileName, int numData) {
    fstream outputDataFile, inputDataFile;
//    /home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte
//    string outputDataFileName = "/home/tri/Desktop/ann_implement/data/sinData.txt";
//    string inputDataFileName = "/home/tri/Desktop/ann_implement/data/sinInput.txt";
    outputDataFile.open(outputDataFileName, ios::out);
    inputDataFile.open(inputDataFileName, ios::out);
    for(int i=0; i <numData; i++){
        // ---- gen true data
        //generate input for sin function by
        // gen an input use uniform distribution + an err use gaussian distribution
        double x = uniformRandom(0, 2*PI);
        double sinX = sin(x)+ gaussianRamdom(0, 0.5) * 0.001;
        //save in file
        inputDataFile<<x<<"\n";
        outputDataFile<<sinX<<"\n";
    }

    outputDataFile.close();
    inputDataFile.close();
}

double DataGenerator::uniformRandom(double floor, double ceil){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(floor, ceil);
    return dis(gen);
}

double DataGenerator::gaussianRamdom(double mean, double neighbour){
    std::random_device rd;
    std::mt19937 gen(rd());

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<> d(mean, neighbour);
    return d(gen);
}