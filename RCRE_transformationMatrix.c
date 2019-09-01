//
// Created by Mads Bjoern on 2019-07-14.
//

#include "RCRE_transformationMatrix.h"
#include "RCRE_constants.h"

PRE_DEVICE RCRE_transformationMatrix *RCRE_transformationMatrix_getTransformationMatrixFromValues(double a, double b, double c,
                                                                                       double d, double e, double f,
                                                                                       double g, double h, double i) {
    RCRE_transformationMatrix *transformationMatrix = (RCRE_transformationMatrix *)malloc(sizeof(RCRE_transformationMatrix));

    transformationMatrix->a = a;
    transformationMatrix->b = b;
    transformationMatrix->c = c;

    transformationMatrix->d = d;
    transformationMatrix->e = e;
    transformationMatrix->f = f;

    transformationMatrix->g = g;
    transformationMatrix->h = h;
    transformationMatrix->i = i;

    return transformationMatrix;
}

PRE_DEVICE RCRE_transformationMatrix *RCRE_transformationMatrix_getIdentityMatrix() {
    return RCRE_transformationMatrix_getTransformationMatrixFromValues( 1, 0, 0,
                                                                        0, 1, 0,
                                                                        0, 0, 1);
}

PRE_DEVICE RCRE_transformationMatrix *RCRE_transformationMatrix_getScalingMatrix(double x, double y, double z) {
    return RCRE_transformationMatrix_getTransformationMatrixFromValues( x, 0, 0,
                                                                        0, y, 0,
                                                                        0, 0, z);
}

PRE_DEVICE RCRE_transformationMatrix *RCRE_transformationMatrix_getShearingMatrix(double xy, double xz,
                                                                       double yx, double yz,
                                                                       double zx, double zy) {
    return RCRE_transformationMatrix_getTransformationMatrixFromValues(  1, xy, xz,
                                                                        yx,  1, yz,
                                                                        zx, zy,  1);
}

PRE_DEVICE RCRE_transformationMatrix *RCRE_transformationMatrix_getRotationMatrixXAxis(double theta) {
    return RCRE_transformationMatrix_getTransformationMatrixFromValues( 1,          0,           0,
                                                                        0, cos(theta), -sin(theta),
                                                                        0, sin(theta),  cos(theta));
}

PRE_DEVICE RCRE_transformationMatrix *RCRE_transformationMatrix_getRotationMatrixYAxis(double theta) {
    return RCRE_transformationMatrix_getTransformationMatrixFromValues(  cos(theta), 0, sin(theta),
                                                                                  0, 1,          0,
                                                                        -sin(theta), 0, cos(theta));
}

PRE_DEVICE RCRE_transformationMatrix *RCRE_transformationMatrix_getRotationMatrixZAxis(double theta) {
    return RCRE_transformationMatrix_getTransformationMatrixFromValues( cos(theta), -sin(theta), 0,
                                                                        sin(theta),  cos(theta), 0,
                                                                                 0,           0, 1);
}


PRE_DEVICE void RCRE_transformationMatrix_multiplication(RCRE_transformationMatrix *m1, RCRE_transformationMatrix *m2, RCRE_transformationMatrix *out) {
    double newA = m1->a * m2->a + m1->b * m2->d + m1->c * m2->g;
    double newB = m1->a * m2->b + m1->b * m2->e + m1->c * m2->h;
    double newC = m1->a * m2->c + m1->b * m2->f + m1->c * m2->i;

    double newD = m1->d * m2->a + m1->e * m2->d + m1->f * m2->g;
    double newE = m1->d * m2->b + m1->e * m2->e + m1->f * m2->h;
    double newF = m1->d * m2->c + m1->e * m2->f + m1->f * m2->i;

    double newG = m1->g * m2->a + m1->h * m2->d + m1->i * m2->g;
    double newH = m1->g * m2->b + m1->h * m2->e + m1->i * m2->h;
    double newI = m1->g * m2->c + m1->h * m2->f + m1->i * m2->i;

    out->a = newA;
    out->b = newB;
    out->c = newC;

    out->d = newD;
    out->e = newE;
    out->f = newF;

    out->g = newG;
    out->h = newH;
    out->i = newI;
}

PRE_DEVICE double RCRE_transformationMatrix_determinant(RCRE_transformationMatrix *m) {
    return  m->a * (m->e * m->i - m->f * m->h) -
            m->b * (m->d * m->i - m->g * m->f) +
            m->c * (m->d * m->h - m->e * m->g);
}

PRE_DEVICE void RCRE_transformationMatrix_inverse(RCRE_transformationMatrix *m, RCRE_transformationMatrix *out) {
    double A =  (m->e * m->i - m->f * m->h), B = -(m->b * m->i - m->c * m->h), C =  (m->b * m->f - m->c * m->e);
    double D = -(m->d * m->i - m->f * m->g), E =  (m->a * m->i - m->c * m->g), F = -(m->a * m->f - m->c * m->d);
    double G =  (m->d * m->h - m->e * m->g), H = -(m->a * m->h - m->b * m->g), I =  (m->a * m->e - m->b * m->d);

    double factor = 1. / RCRE_transformationMatrix_determinant(m);

    out->a = A * factor;
    out->b = B * factor;
    out->c = C * factor;

    out->d = D * factor;
    out->e = E * factor;
    out->f = F * factor;

    out->g = G * factor;
    out->h = H * factor;
    out->i = I * factor;
}

PRE_DEVICE void RCRE_transformationMatrix_applyToPoint(RCRE_transformationMatrix *m, RCRE_point3D *p, RCRE_point3D *out) {
    double newX = m->a * p->x + m->b * p->y + m->c * p->z;
    double newY = m->d * p->x + m->e * p->y + m->f * p->z;
    double newZ = m->g * p->x + m->h * p->y + m->i * p->z;

    out->x = newX;
    out->y = newY;
    out->z = newZ;
}

PRE_DEVICE void RCRE_transformationMatrix_applyToPointWithRegardsToPoint(RCRE_transformationMatrix *m, RCRE_point3D *p,
                                                              RCRE_point3D *o, RCRE_point3D *out) {
    RCRE_point3D po;
    RCRE_point3D_subtract(p, o, &po);
    RCRE_transformationMatrix_applyToPoint(m, &po, &po);
    RCRE_point3D_add(&po, o, out);
}
