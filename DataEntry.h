//
// Created by tri on 18/06/2017.
//

#ifndef FAKE_CODE_DATAENTRY_H
#define FAKE_CODE_DATAENTRY_H
#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

class DataEntry
{
public:

    //public members
    Mat pattern;	//all the patterns
    double target;		//all the targets

public:

    //constructor

    DataEntry(Mat p, double t): pattern(p), target(t) {}

    ~DataEntry() {

    }

};
#endif //FAKE_CODE_DATAENTRY_H
