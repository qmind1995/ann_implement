//
// Created by tri on 18/07/2017.
//

#include "HOGFeature.h"
#include <math.h>

// in this time, i just implement hog apply some simple kernel. I'll update later.
HOGFeature::HOGFeature(int bin_, int cellSize_, int blockSize_):bin(bin_), cellSize(cellSize_), blockSize(blockSize_) {
    overlap = 1;
}

mat HOGFeature::featureDetect(mat image) {
    mat kernel(1,3);
    kernel(0,0) = -1;
    kernel(0,1) = 0;
    kernel(0,2) = 1;

    mat gradientX, gradientY;

    gradientX = convolution(image, kernel);
    gradientY = convolution(image, kernel.t());

    mat magnitude = mat(image.n_rows, image.n_cols);
    magnitude.zeros();
    mat orientation = mat(image.n_rows, image.n_cols);
    orientation.zeros();
    Mat<int> orientBin(image.n_rows, image.n_cols);
    orientBin.zeros();

    double epsilon = pow(10, -5);
    for(int i=0; i< image.n_rows; i++){
        for(int j=0; j< image.n_cols; j++){
            magnitude(i,j) = sqrt(pow(gradientX(i,j),2) + pow(gradientY(i,j),2) ) / (255+ 255);
            orientation(i,j) = atan(gradientX(i,j) / (gradientY(i,j) + epsilon)) + PI/2; //plus Pi/2 to change [-PI/2 - PI/2] -> [0-PI]
            orientBin(i,j) = (int)((int)round(orientation(i, j) * bin/ PI) % bin);
        }
    }

    int ncellRow = (image.n_rows / cellSize),
        ncellCol = image.n_cols / cellSize;

    cube cellVote((const uword) ncellRow, (const uword) ncellRow, (const uword) bin);
    cellVote.zeros();

    for(int i = 0; i < image.n_rows; i++){

        int icellRow = (int)floor(i / cellSize);
        for(int j = 0; j < image.n_cols; j++){

            int icellCol = (int)floor(j / cellSize);
            int voteIndex = orientBin(i, j);

            cellVote(icellRow, icellCol, voteIndex) += magnitude(i, j);
        }
    }

    int nBlockR = ( (image.n_rows - blockSize*cellSize) / cellSize +1 );
    int nBlockC = ( (image.n_cols - blockSize*cellSize) / cellSize +1 );
    int nfeatureBlock = blockSize*blockSize * bin;
    int nfeature = nBlockR *  nBlockC* nfeatureBlock;

    cube blockFeature((const uword) nBlockR, (const uword) nBlockC, (const uword) nfeatureBlock);
    blockFeature.zeros();

    for(int i=0; i< nBlockR; i++){
        for(int j=0; j< nBlockC; j++){

            //for each cell in block
            for(int a =0; a< blockSize; a++){

                int cellIndexR = i * (blockSize-overlap) + a;

                for(int b= 0; b< blockSize; b++){

                    int cellIndexC = j * (blockSize-overlap) + b;

                    for(int c = 0; c < bin; c++){

                        int fIndex = (a+1) * b *bin + c;
                        blockFeature(i, j, fIndex) = cellVote(cellIndexR, cellIndexC, c);
                    }

                }
            }

        }
    }

    mat feature(nfeature,1);
    feature.zeros();

    for(int i=0; i< nBlockR; i++) {
        for (int j = 0; j < nBlockC; j++) {
            for(int k =0 ;k < nfeatureBlock; k++){
                int fIndex = (i + 1)* j * nfeatureBlock +k;
                feature(fIndex, 0) = blockFeature(i, j, k);
            }
        }
    }

    return feature;
}

mat HOGFeature::convolution(mat image, mat kernel) {
    mat image_(image.n_rows, image.n_cols);

    //flip kernal
    int middle_r = (int)std::ceil((double)kernel.n_rows / 2.0);
    int middle_c = (int)std::ceil((double)kernel.n_cols/ 2.0);
    if(middle_r > 1){
        for(int i=0; i< middle_r; i++){
            for(int j=0; j< kernel.n_cols; j++){
                auto tmp = kernel(i, j);
                kernel(i, j) = kernel(kernel.n_rows - 1 -i, j);
                kernel(kernel.n_rows - 1 -i, j) = tmp;
            }
        }
    }
    if(middle_c > 1){
        for(int i=0; i< middle_c; i++){
            for(int j=0; j< kernel.n_rows; j++){
                auto tmp = kernel(j,i);
                kernel(j,i) = kernel(j, kernel.n_cols - 1 -i);
                kernel(j, kernel.n_cols - 1 -i) = tmp;
            }
        }
    }


    int paddingH = kernel.n_cols - middle_c;
    int paddingV = kernel.n_rows - middle_r;

    // assume both paddingV and paddingH < 1

    for(int i =0; i <image.n_rows; i++){
        for(int j =0 ;j <image.n_cols; j++){

            double gd = 0.0;

            for(int v = -paddingV; v <= paddingV; v++){
                for(int h = -paddingH; h <= paddingH; h++){

                    if(i+v >0 && i+v < image.n_rows && j+h > 0 && j+h < image.n_cols){
                        image_(i, j) += kernel(v + paddingV, h + paddingH) * image(i+v, j+h);
                    }

                }
            }

        }
    }
    return image_;
}