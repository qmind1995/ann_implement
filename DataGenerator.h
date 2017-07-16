//
// Created by tri on 15/07/2017.
//

#ifndef FAKE_CODE_DATAGENERATOR_H
#define FAKE_CODE_DATAGENERATOR_H

#include "DataEntry.h"
#include <math.h>

#define PI 3.141592654


class DataGenerator{
public:
    vector<DataEntry*> getDataForReg();
    void genDataForSinFunction(int numData);
private:
    double uniformRandom(double floor, double ceil);
    double gaussianRamdom(double floor, double ceil);
};



#endif //FAKE_CODE_DATAGENERATOR_H
