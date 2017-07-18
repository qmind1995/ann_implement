//
// Created by tri on 18/07/2017.
//

#include "HOGFeature.h"
#include <math.h>

// in this time, i just implement hog apply some simple kernel. I'll update later.
HOGFeature::HOGFeature(int bin_, int cellSize_, int blockSize_):bin(bin_), cellSize(cellSize_), blockSize(blockSize_) {
}

mat HOGFeature::featureDetect(mat image) {
    mat feature;
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

    double epsilon = pow(10, -5);
    for(int i=0; i< image.n_rows; i++){
        for(int j=0; j< image.n_cols; j++){
            magnitude(i,j) = sqrt(pow(gradientX(i,j),2) + pow(gradientY(i,j),2) );
            orientation(i,j) = atan(gradientX(i,j) / (gradientY(i,j) + epsilon)) / PI *180;
        }
    }



    int ncellRow = (image.n_rows / cellSize),
        ncellCol = image.n_cols / cellSize;




    cout<<orientation<<endl<<magnitude<<endl;

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