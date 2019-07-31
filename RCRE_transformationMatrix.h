//
// Created by Mads Bjoern on 2019-07-14.
//

#include "RCRE_point3D.h"

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_TRANSFORMATIONMATRIX_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_TRANSFORMATIONMATRIX_H

struct RCRE_transformationMatrix {
    double  a, b, c,
            d, e, f,
            g, h, i;
} typedef RCRE_transformationMatrix;


RCRE_transformationMatrix *RCRE_transformationMatrix_getTransformationMatrixFromValues(double a, double b, double c,
                                                                                       double d, double e, double f,
                                                                                       double g, double h, double i);
RCRE_transformationMatrix *RCRE_transformationMatrix_getIdentityMatrix();
RCRE_transformationMatrix *RCRE_transformationMatrix_getScalingMatrix(double x, double y, double z);
RCRE_transformationMatrix *RCRE_transformationMatrix_getShearingMatrix(double xy, double xz,
                                                                       double yx, double yz,
                                                                       double zx, double zy);
RCRE_transformationMatrix *RCRE_transformationMatrix_getRotationMatrixXAxis(double theta);
RCRE_transformationMatrix *RCRE_transformationMatrix_getRotationMatrixYAxis(double theta);
RCRE_transformationMatrix *RCRE_transformationMatrix_getRotationMatrixZAxis(double theta);

void RCRE_transformationMatrix_multiplication(RCRE_transformationMatrix *m1, RCRE_transformationMatrix *m2, RCRE_transformationMatrix *out);
double RCRE_transformationMatrix_determinant(RCRE_transformationMatrix *m);
void RCRE_transformationMatrix_inverse(RCRE_transformationMatrix *m, RCRE_transformationMatrix *out);
void RCRE_transformationMatrix_applyToPoint(RCRE_transformationMatrix *m, RCRE_point3D *p, RCRE_point3D *out);
void RCRE_transformationMatrix_applyToPointWithRegardsToPoint(RCRE_transformationMatrix *m, RCRE_point3D *p, RCRE_point3D *o, RCRE_point3D *out);

#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_TRANSFORMATIONMATRIX_H
