//
// Created by tri on 23/07/2017.
//

#include <GL/glut.h>

void draw(void) {

    // Black background
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //Draw i
    glFlush();
    //https://cmake.org/pipermail/cmake/2009-February/027234.html

}
//
//int main(int argc, char **argv) {
//    // test hog
//    glutInit(&argc, argv);
//    glutInitDisplayMode(GLUT_RGBA|GLUT_SINGLE);
//    glutInitWindowPosition(50, 25);
//
//    //Configure Window Size
//    glutInitWindowSize(480,480);
//
//    //Create Window
//    glutCreateWindow("Hello OpenGL");
//
//
//    //Call to the drawing function
//    glutDisplayFunc(draw);
//
//    // Loop require by OpenGL
//    glutMainLoop();
//    return 0;
//
//
//}
