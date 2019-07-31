//
// Created by Mads Bjoern on 2019-07-09.
//

#ifndef RAYCASTINGRENDERINGENGINEGPU_RCRE_TRIANGLE3D_H
#define RAYCASTINGRENDERINGENGINEGPU_RCRE_TRIANGLE3D_H

#include "RCRE_point3D.h"

struct RCRE_triangle3D {
    RCRE_point3D *p1, *p2, *p3;
} typedef RCRE_triangle3D;

RCRE_triangle3D *RCRE_triangle3D_getTriangleFromPoints(RCRE_point3D *p1, RCRE_point3D *p2, RCRE_point3D *p3);
RCRE_triangle3D *RCRE_triangle3D_getTriangleFromPointValues(RCRE_point3D *p1, RCRE_point3D *p2, RCRE_point3D *p3);

bool RCRE_triangle3D_equals(RCRE_triangle3D *t1, RCRE_triangle3D *t2);

// if there is any combination that makes the triangles equal
bool RCRE_triangle3D_equalsWeak(RCRE_triangle3D *t1, RCRE_triangle3D *t2);

double RCRE_triangle3D_getArea(RCRE_triangle3D *t);
double RCRE_triangle3D_getAreaTimes2(RCRE_triangle3D *t);

void RCRE_triangle3D_rotateTriangleAfterPoint(RCRE_triangle3D *t, RCRE_point3D *p);

// used to distinguish between which of the 2 sides a point lies, compare to the triangle
// 1, 0, -1 (only return 0 is one the same plane as t)
int RCRE_triangle3D_isPointOnPositiveSide(RCRE_triangle3D *t, RCRE_point3D *p);

bool RCRE_triangle3D_getIntersectionPoint(RCRE_triangle3D *t, RCRE_point3D *rayOrigin, RCRE_point3D *rayDirection, RCRE_point3D *outIntersectionPoint, RCRE_point3D *outReflectiveDirection);

#endif //RAYCASTINGRENDERINGENGINEGPU_RCRE_TRIANGLE3D_H
