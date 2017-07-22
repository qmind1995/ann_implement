//
// Created by tri on 22/07/2017.
//

#include "armadillo"

using namespace std;
using namespace arma;

inline static mat dotProduct(mat A, mat B){

    auto sizeA = arma::size(A);
    auto sizeB = arma::size(B);
    if(sizeA.n_rows != sizeB.n_rows || sizeA.n_cols != sizeB.n_cols){
        cout<<"dont match size A and B ! check again idot!"<<endl;
    }
    mat output = mat(sizeA.n_rows, sizeA.n_cols);
    for(int i =0; i< sizeA.n_rows; i++){
        for(int j=0; j< sizeA.n_cols; j++){
            output(i,j) = A(i,j) * B(i,j);
        }
    }
    return output;
}


inline static double uniformRandom(double floor, double ceil){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(floor, ceil);
    return dis(gen);
}

inline static double gaussianRamdom(double mean, double neighbour){
    std::random_device rd;
    std::mt19937 gen(rd());

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<> d(mean, neighbour);
    return d(gen);
}
