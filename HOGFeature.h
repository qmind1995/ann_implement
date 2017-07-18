//
// Created by tri on 18/07/2017.
//

#ifndef FAKE_CODE_HOGFEATURE_H
#define FAKE_CODE_HOGFEATURE_H

#include "DataEntry.h"


#include <armadillo>
using namespace std;
using namespace arma;

#define PI 3.141592654

class HOGFeature {
public:
    HOGFeature(int bin_, int cellSize_, int blockSize_);
    mat featureDetect(mat image);

private:
    int bin;
    int cellSize;
    int blockSize;
    int overlap;

    mat convolution(mat image, mat kernel);

};
#endif //FAKE_CODE_HOGFEATURE_H
