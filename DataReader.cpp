//
// Created by tri on 17/06/2017.
//

//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <math.h>
#include "DataReader.h"

using namespace arma;
//using namespace cv;
using namespace std;

DataReader::~DataReader() {
    //clear data
//    for (int i=0; i < (int) data.size(); i++ ) delete data[i];
//    data.clear();
}

int DataReader::ReverseInt (int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void DataReader::read_Mnist(string filename, vector<arma::mat> &vec){
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
        for(int i = 0; i < number_of_images; ++i) {
            arma::mat tp(n_rows*n_cols,1);

            for(int r = 0; r < n_rows; ++r) {
                for(int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp((r+1)*c, 0) = (double) temp;
                }
            }
            vec.push_back(tp);
        }

    }
}
//void DataReader::read_Mnist(string filename, vector<cv::Mat> &vec){
//    ifstream file(filename,ios::in | ios::binary);
//
//    if (file.is_open()) {
//        int magic_number = 0;
//        int number_of_images = 0;
//        int n_rows = 0;
//        int n_cols = 0;
//        file.read((char*) &magic_number, sizeof(magic_number));
//        magic_number = ReverseInt(magic_number);
//        file.read((char*) &number_of_images,sizeof(number_of_images));
//        number_of_images = ReverseInt(number_of_images);
//        file.read((char*) &n_rows, sizeof(n_rows));
//        n_rows = ReverseInt(n_rows);
//        file.read((char*) &n_cols, sizeof(n_cols));
//        n_cols = ReverseInt(n_cols);
//        for(int i = 0; i < number_of_images; ++i) {
//            cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
//            for(int r = 0; r < n_rows; ++r) {
//                for(int c = 0; c < n_cols; ++c) {
//                    unsigned char temp = 0;
//                    file.read((char*) &temp, sizeof(temp));
//                    tp.at<uchar>(r, c) = (int) temp;
//                }
//            }
//            vec.push_back(tp);
//        }
//    }
//    else{
//        cout<<"cannot open file!!!"<<endl;
//    }
//}

void DataReader::read_Mnist_Label(string filename, vector<double> &vec) {
    ifstream file (filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i) {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}

void DataReader::read_Input(string imgFileName, string labelFileName){
    int number_of_images = 10000;
    vector<arma::mat> vecData;
    read_Mnist(imgFileName, vecData);
    vector<double> vecLabel(number_of_images);
    read_Mnist_Label(labelFileName, vecLabel);
    //preprocess data:
    for(int i = 0; i < number_of_images; ++i) {

        arma::mat target = mat(10,1);
        target.zeros();
        for(int idx= 0; idx<10; idx++){
            if(abs(vecLabel[i] - idx) <0.0001){
                target[idx] =1;
                break;
            }
        }

        DataEntry *d = new DataEntry(vecData[i],target);
        data.push_back(d);
    }
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
