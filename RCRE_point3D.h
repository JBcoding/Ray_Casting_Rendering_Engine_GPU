//
// Created by Mads Bjoern on 2019-07-09.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_POINT3D_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_POINT3D_H

#include <stdbool.h>

struct RCRE_point3D {
    double x, y, z;
} typedef RCRE_point3D;

static RCRE_point3D RCRE_point3D_ORIGIN = {.x = 0, .y = 0, .z = 0};

RCRE_point3D *RCRE_point3D_getPointFromValues(double x, double y, double z);
RCRE_point3D *RCRE_point3D_copy(RCRE_point3D *p);
void RCRE_point3D_copyInto(RCRE_point3D *p, RCRE_point3D *out);

bool RCRE_point3D_equal(RCRE_point3D *a, RCRE_point3D *b);
void RCRE_point3D_add(RCRE_point3D *a, RCRE_point3D *b, RCRE_point3D *out);
void RCRE_point3D_subtract(RCRE_point3D *a, RCRE_point3D *b, RCRE_point3D *out);
void RCRE_point3D_scale(RCRE_point3D *p, double scale, RCRE_point3D *out);
double RCRE_point3D_distance(RCRE_point3D *a, RCRE_point3D *b);
double RCRE_point3D_distanceToOrigin(RCRE_point3D *a);
void RCRE_point3D_crossProduct(RCRE_point3D *a, RCRE_point3D *b, RCRE_point3D *out);
double RCRE_point3D_dotProduct(RCRE_point3D *a, RCRE_point3D *b);
void RCRE_point3D_getUnit(RCRE_point3D *p, RCRE_point3D *out);
void RCRE_point3D_rotatePointAroundAxis(RCRE_point3D *p, RCRE_point3D *axisDirection, RCRE_point3D *axisPoint, double angle, RCRE_point3D *out);
double RCRE_point3D_getAngleBetweenPoints(RCRE_point3D *p1, RCRE_point3D *p2);

bool RCRE_point3D_arePointsOnTheSamePlane(RCRE_point3D *a, RCRE_point3D *b, RCRE_point3D *c, RCRE_point3D *d);
bool RCRE_point3D_averagePointsWithWeight(RCRE_point3D *a, RCRE_point3D *b, double weight, RCRE_point3D *out);

#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_POINT3D_H
