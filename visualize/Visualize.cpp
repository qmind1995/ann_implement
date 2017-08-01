//
// Created by tri on 23/07/2017.
//

#include "../ann/NeuralNetwork.h"
#include <GL/glut.h>
#include "../Utils.cpp"
#include "../ann/Trainer.h"

using namespace parameters;
using namespace std;

static NeuralNetwork *visualizeNet ;
static Trainer * trainer;

static void text(string text, int line) {

    int len = (int) text.length();
    glColor3f(1,1,1);

    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glLoadIdentity();

    gluOrtho2D( 0, 600, 0, 600 );

    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();

    glLoadIdentity();

    glRasterPos2i(10, 570 - 20*line);

    for ( int i = 0; i < len; ++i ) {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, text[i]);
    }

    glPopMatrix();

    glMatrixMode( GL_PROJECTION );
    glPopMatrix();
    glMatrixMode( GL_MODELVIEW );
}

static void renderSineWave() {
    glColor3f(1.0,0.0,0.0);
    glPointSize(3);
    glBegin(GL_POINTS);
    for(GLdouble i= 0; i < PI*2; i +=0.01) {
        GLdouble x = i* 180 / PI;
        GLdouble y = 100.0 * sin(i);
        glVertex2d(x, y);
    }
    glEnd();
}

static void renderRegression(int nPoints){
    glColor3f(1.0,1.0,0.0);
//    double x = uniformRandom(0, 2*PI);
    glBegin(GL_POINTS);
    if(visualizeNet != NULL){
        for(int i= 0; i < nPoints; i++) {
            GLdouble x = uniformRandom(0, 2*PI);
            mat input = mat(1,1);
            input(0,0) = x;
            mat output = visualizeNet->getVisualizeOutput(input);
            GLdouble y = output(0,0);
            glVertex2d(x*180/PI, y*100);
        }
    }
    glEnd();
}

static void display(){
    glClearColor(0.0, 0.0, 0.0, 1.0);  // clear background with black
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    double w = glutGet( GLUT_WINDOW_WIDTH );
    double h = glutGet( GLUT_WINDOW_HEIGHT );
    double ar = w / h;
    glOrtho( -120 , 360 * ar, -120, 120, -1, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    // call draw func:
    renderSineWave();
    renderRegression(1000);

    glutSwapBuffers();
}

static void renderNetworkInfo(){
    glClearColor((GLclampf) (100.0 / 255), (GLclampf) (100.0 / 255), (GLclampf) (100.0 / 255), 1.0);  // clear background
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    double w = glutGet( GLUT_WINDOW_WIDTH );
    double h = glutGet( GLUT_WINDOW_HEIGHT );
    double ar = w / h;
    glOrtho( -120 , 360 * ar, -120, 120, -1, 1 );
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    // call draw func:
    if(visualizeNet != NULL){
        auto info = visualizeNet->getNeuralInfoForVisualize();
        int nline = (int) info.size();
        for(int i=0; i < nline; i++){
            text(info[i], i);
        }

        text("training set accuracy: " + to_string(trainer->trainingSetMSE), nline + 1);
    }

    glutSwapBuffers();
}

static void processTimer(int value){

    glutTimerFunc(100, processTimer, value);
    glutPostRedisplay();
}

static void resize(int width, int height) {

    glutReshapeWindow( width, height);
}

static void visualize(Trainer * tn, NeuralNetwork * nnet, int argc, char** argv) {
    visualizeNet = nnet;
    trainer = tn;
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( 1100, 480);
    int mainWindow = glutCreateWindow( "SineWave" );
    glutTimerFunc(50, processTimer, 1);
    glutDisplayFunc( display );

    // init subwindow
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // register callbacks
    glutIgnoreKeyRepeat(1);
    glutCreateSubWindow(mainWindow, 680 ,0,1100-680,480);
    glutDisplayFunc(renderNetworkInfo);
    glutReshapeFunc(resize);
    //http://www.lighthouse3d.com/tutorials/glut-tutorial/subwindow-reshape/

    glutMainLoop();
}


