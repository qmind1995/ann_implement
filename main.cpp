//
// Created by tri on 16/06/2017.
//
#include<iostream>
#include "NeuralNetwork.h"
#include "Trainer.h"
#include "BatchTrainer.h"
#include "DataGenerator.h"

#include <GL/glut.h>

using namespace std;
using namespace arma;

/*
// test openGL
void draw(void) {

    // Black background
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //Draw i
    glFlush();
    //https://cmake.org/pipermail/cmake/2009-February/027234.html


}

int main(int argc, char **argv) {
    // test hog
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA|GLUT_SINGLE);
    glutInitWindowPosition(50, 25);

    //Configure Window Size
    glutInitWindowSize(480,480);

    //Create Window
    glutCreateWindow("Hello OpenGL");


    //Call to the drawing function
    glutDisplayFunc(draw);

    // Loop require by OpenGL
    glutMainLoop();
    return 0;


}

*/


int main(){

    int isMnist = 0;

    cout<<endl<<endl
        <<"PRESS 1 TO ACCESS MNIST LEARNING, 0 TO ACCESS SIN METHOD LEARNING"<<endl;
    cin>>isMnist;

    if(isMnist == 1){

        NeuralNetwork *nn = new NeuralNetwork(784,100,10, "SIGMOID");
        //NeuralNetwork *nn = new NeuralNetwork(1296,100,10, "SIGMOID");
        //NeuralNetwork *nn = new NeuralNetwork("weights.txt","SIGMOID");
        string imgFileName_test = "/home/tri/Desktop/ann_implement/data/t10k-images.idx3-ubyte";
        string labelFileName_test = "/home/tri/Desktop/ann_implement/data/t10k-labels.idx1-ubyte";
        string imgFileName = "/home/tri/Desktop/ann_implement/data/train-images.idx3-ubyte";
        string labelFileName = "/home/tri/Desktop/ann_implement/data/train-labels.idx1-ubyte";
        DataReader *dR = new DataReader();
        dR->read_Input(imgFileName, labelFileName, 60000);
        srand( (unsigned int) time(0) );

        trainingDataSet* trSet = new trainingDataSet();
        trSet->trainingSet = dR->data;

        DataReader *dR_test = new DataReader();
        dR_test->read_Input(imgFileName_test, labelFileName_test, 10000);
        trSet->validationSet = dR_test->data;

        //create neural network trainer
        int isBatch = 0 ;
        cout<<"PRESS 1 TO TRAIN BY BATCH, 0 TO TRAIN BY EACH MODEL"<<endl;
        cin>>isBatch;
        if(isBatch == 1){
            BatchTrainer nT(nn, 100);
            nT.trainNetwork(trSet);
        }
        else{
            Trainer nT( nn );
            nT.trainNetwork(trSet);
        }
    }
    else{
        int shouldGenFile = false;
        string outputDataFileName = "/home/tri/Desktop/ann_implement/data/sinData.txt";
        string outputTestFileName = "/home/tri/Desktop/ann_implement/data/sinData_test.txt";
        string inputDataFileName = "/home/tri/Desktop/ann_implement/data/sinInput.txt";
        string inputtestFileName = "/home/tri/Desktop/ann_implement/data/sinInput_test.txt";

        cout<<"If you want to gen data files for sin method, press 1! else press 0."<<endl;
        cin>>shouldGenFile;

        if(shouldGenFile == 1){
            DataGenerator *dataGen = new DataGenerator();
            dataGen->genDataForSinFunction(inputDataFileName, outputDataFileName, 60000);
            dataGen->genDataForSinFunction(inputtestFileName, outputTestFileName, 10000);
            cout<<"gen data done!"<<endl;
        }

        DataReader *dR = new DataReader();
        dR->read_RegressionData(inputDataFileName, outputDataFileName, 10000);

        NeuralNetwork *nn = new NeuralNetwork(1,20,1, "TANH");

        Trainer nT( nn );

        trainingDataSet* trSet = new trainingDataSet();
        trSet->trainingSet = dR->data;

        DataReader *dR_test = new DataReader();
        dR_test->read_RegressionData(inputtestFileName, outputTestFileName, 3000);
        trSet->validationSet = dR_test->data;

        nT.trainNetwork(trSet);
    }

    return 0;
}

