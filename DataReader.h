//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_DATAREADER_H
#define ANN_IMPLEMENT_DATAREADER_H

#include "DataEntry.h"
#include <armadillo>


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
    int trainingDataEndIndex;

//public methods
public:

    DataReader(): creationApproach(NONE), numTrainingSets(-1) {}
    ~DataReader();
//    void read_Mnist(string filename, vector<cv::Mat> &vec);
    void read_Mnist(string filename, vector<arma::mat> &vec);
    void read_Mnist_Label(string filename, vector<double> &vec);
    void read_Input(string imgFileName, string labelFileName);
//private methods
private:
    int ReverseInt (int i);

};
#endif //ANN_IMPLEMENT_DATAREADER_H
