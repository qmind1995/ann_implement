//
// Created by tri on 23/07/2017.
//

#include "../ann/NeuralNetwork.h"
#include <GL/glut.h>

using namespace parameters;

static NeuralNetwork *visualizeNet ;

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

static void renderRegression(){
    glColor3f(1.0,1.0,0.0);

    glBegin(GL_POINTS);
    for(GLdouble i= 0; i < PI*2; i +=0.01) {
        GLdouble x = i* 180 / PI;
        mat input = mat(1,1);
        input(0,0) = i;

        mat output = visualizeNet->getVisualizeOutput(input);
        GLdouble y = output(0,0) * 100.0;
        glVertex2d(x, y);
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
    renderRegression();

    glutSwapBuffers();
}

static void processTimer(int value){

    glutTimerFunc(50, processTimer, value);
    glutPostRedisplay();
}

static void visualize(NeuralNetwork * nnet, int argc, char** argv) {
    visualizeNet = nnet;
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( 640, 480);
    glutCreateWindow( "SineWave" );
    glutTimerFunc(50, processTimer, 1);
    glutDisplayFunc( display );
    glutMainLoop();
}


