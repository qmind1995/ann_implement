//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_DATAREADER_H
#define ANN_IMPLEMENT_DATAREADER_H

#include "DataEntry.h"
#include <armadillo>


class trainingDataSet {
public:

    std::vector<DataEntry*> trainingSet;
    std::vector<DataEntry*> generalizationSet;
    std::vector<DataEntry*> validationSet;

    trainingDataSet(){}

    void clear() {
        trainingSet.clear();
        generalizationSet.clear();
        validationSet.clear();
    }
};


enum { NONE, STATIC, GROWING, WINDOWING };

class DataReader {
    //public members
public:
    std::vector<DataEntry*> data;
//http://eric-yuan.me/cpp-read-mnist/
//private members
private:

    //data storage

    int nInputs;
    int nTargets;

    //data set creation approach and total number of dataSets
    int creationApproach;
    int numTrainingSets;

//public methods
public:

    DataReader(): creationApproach(NONE), numTrainingSets(-1) {}
    ~DataReader();
    void read_Mnist(string filename, vector<arma::mat> &vec, int max_number_of_images);
    void read_Mnist_Label(string filename, vector<double> &vec, int max_number_of_images);
    void read_Input(string imgFileName, string labelFileName, int number_of_images);
    void read_RegressionData(string inputFileName, string outputFileName, int numdata);
    void read_Mnist_HOG(string filename, vector<arma::mat> &vec, int max_number_of_images);
//private methods
private:
    int ReverseInt (int i);

};
#endif //ANN_IMPLEMENT_DATAREADER_H
