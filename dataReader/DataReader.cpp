//
// Created by tri on 17/06/2017.
//


#include <iostream>
#include <fstream>
#include <math.h>
#include "DataReader.h"
#include "HOGFeature.h"

using namespace arma;
using namespace std;

DataReader::~DataReader() {}

int DataReader::ReverseInt (int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void DataReader::read_Mnist(string filename, vector<arma::mat> &vec, int max_number_of_images){
    ifstream file(filename,ios::in | ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < max_number_of_images; ++i) {
            arma::mat tp(n_rows*n_cols,1);

            for(int r = 0; r < n_rows; ++r) {
                for(int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp((r+1)*c, 0) = (double) temp /255;
                }
            }
            vec.push_back(tp);
        }

    }
}

void DataReader::read_Mnist_Label(string filename, vector<double> &vec, int max_number_of_images) {
    ifstream file (filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images =0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < max_number_of_images; ++i) {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}

void DataReader::read_Input(string imgFileName, string labelFileName, int number_of_images){
    vector<arma::mat> vecData;
    read_Mnist(imgFileName, vecData, number_of_images);
//    read_Mnist_HOG(imgFileName, vecData, number_of_images);
    vector<double> vecLabel(number_of_images);
    read_Mnist_Label(labelFileName, vecLabel, number_of_images);
    //preprocess data:
    for(int i = 0; i < number_of_images; ++i) {

        arma::mat target = mat(10,1);
        target.zeros();
        for(int idx= 0; idx < 10; idx++){
            if(abs(vecLabel[i] - idx) <0.0001){
                target[idx] =1;
                break;
            }
        }

        DataEntry *d = new DataEntry(vecData[i],target);
        data.push_back(d);
    }
}

void DataReader::read_RegressionData(string inputFileName, string outputFileName, int numdata) {
    fstream inputFile, outputFile;
    inputFile.open(inputFileName, ios::in);
    outputFile.open(outputFileName, ios::in);
    string inputLine, outputLine;
    for(int i=0; i<numdata; i++){
        getline(inputFile, inputLine);
        getline(outputFile, outputLine);
        double input = std::stod(inputLine);
        double output = std::stod(outputLine);
        mat iD = mat(1,1);
        iD(0,0) = input;
        mat oD = mat(1,1);
        oD(0,0) = output;
        DataEntry *d = new DataEntry(iD,oD);
        data.push_back(d);
    }

}

void DataReader::read_Mnist_HOG(string filename, vector<arma::mat> &vec, int max_number_of_images) {

    ifstream file(filename,ios::in | ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        HOGFeature *hf = new HOGFeature(9, 4, 2);

        for(int i = 0; i < max_number_of_images; ++i) {
            arma::mat tp(n_rows,n_cols);

            for(int r = 0; r < n_rows; ++r) {
                for(int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp(r,c) = (double) temp ;
                }
            }

            mat hogFeature =  hf->featureDetect(tp);
            vec.push_back(hogFeature);
        }
    }

//    hf->featureDetect()
}

//int main() {
//    string imgFileName = "/home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte";
//    string labelFileName = "/home/tri/Desktop/ann_implement/data/t10k-labels.idx1-ubyte";
//
//    DataReader *dR = new DataReader();
//    dR->read_Input(imgFileName, labelFileName);
//    cout <<dR->data[0]->pattern;
//return 0;
//}
